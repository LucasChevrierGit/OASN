from matplotlib.style import available
from numpy import diff
from humanAgent import HumanAgent
import asyncio
import json
from openai import OpenAI
import random
import matplotlib.pyplot as plt

import enum
import pandas as pd
import os

class Difficulty(enum.Enum):
    easy = 0
    medium = 1
    hard = 2

class GameStage(enum.Enum):
    first_choice = 1
    advised_choice = 2
    neighbors_choice = 3

client = OpenAI()

advised_choice_text = open("prompts/advised_choice.txt", "r").read()
neighbors_choice_text = open("prompts/neighbors_choice.txt", "r").read()
first_choice_text = open("prompts/first_choice.txt", "r").read()

class ExperimentGroup:
    def __init__(self, nb_rounds, nb_agents, pre_prompts, max_neighbor=3, initial_network=None, shockRate=0.1):
        self._game_id = random.randint(0, 1000000)
        self.nb_rounds = nb_rounds
        self.max_neighbor = max_neighbor
        self.nb_agents = nb_agents
        self._client = client
        self._shockRate = shockRate
        if self.max_neighbor >= self.nb_agents:
            raise ValueError("Number of neighbors must be less than number of agents")
        
        self.game = {
            i: {
            j: {
                stage: {} for stage in GameStage
            } for j in range(self.nb_agents)
            } for 
            i in range(self.nb_rounds)
        }
        self.agents = self._generateAgent(pre_prompts)
        if initial_network and len(initial_network) == self.nb_agents:
            for agent in self.agents:
                agent.neigbors = initial_network[agent.id]
        else:
            self._set_neigbors_randomly()
        self.tasks = json.load(open("task/recap.json"))
                
        self._correlation_tool = [
            {
                "type": "function",
                "function": {
                    "name": "correlation_score",
                    "description": "Determine the correlation strength of the given scatter plot by selecting an integer between 0 and 100.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selected_value": {
                                "type": "integer",
                                "description": "The chosen integer value.",
                                "minimum": 0,
                                "maximum": 100
                            }
                        },
                        "required": ["selected_value"],
                    },
                }
            }
        ]

    def _get_neigbors_tool(self, agent_id, nb_neighbors):
        available_neighbors = [agent.id for agent in self.agents if agent.id != agent_id]
        return [
            {
                "type": "function",
                "function": {
                    "name": "choose_multiple_integers",
                    "description": "Select up to {nb_neighbors} participant IDs to follow in the next round.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selected_value": {
                                "type": "array",
                                "items": {
                                    "type": "integer",
                                    "enum": available_neighbors
                                },
                                "description": (
                                f"An array of up to {nb_neighbors} integers representing "
                                f"the participant IDs that should be followed."
                            ),
                            }
                        },
                        "required": ["selected_value"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]

    def _generateAgent(self, preprompts):
        if len(preprompts) == self.nb_agents:
            return [HumanAgent(client=self._client, id = i, preprompt= preprompts[i] , difficulty= Difficulty(i%3).name) for i in range(len(preprompts))]
        else:
            raise ValueError("chelou")

        
    async def run_first_stage(self, task, round, is_print=False):
        for agent in self.agents:
            text = first_choice_text
            agent.add_image_to_discussion(task["difficultyPath"][agent.difficulty][1:], text)
        
        response_tasks = [agent.get_answer(tools= self._correlation_tool, tool_choice="auto") for agent in self.agents]
        responses = await asyncio.gather(*response_tasks)
        for message_content, choice, agent_id  in responses:
                self.game[round][agent_id][GameStage.first_choice]["choice"] = choice
                self.game[round][agent_id][GameStage.first_choice]["content"] = message_content


    async def run_advised_choice(self, round, is_print=False, times=3):
        for i in range(times):
            for agent in self.agents:
                if i==0:
                    neighbor_responses = "\n".join([f"- Participant {neighbor_id} chose the value {self.game[round][neighbor_id][GameStage.first_choice]['choice']}" for neighbor_id in agent.neigbors])
                    text = f"Thank you for your answer, here are the first answer of the participants you chose to follow :\n {neighbor_responses}\n\n" + advised_choice_text
                else:
                    neighbor_responses = "\n".join([f"- Participant {neighbor_id} chose the new value {neighbor_id} : {self.game[round][neighbor_id][GameStage.advised_choice]['choice'][-1]}" for neighbor_id in agent.neigbors])
                    text = f"Upon reflection, just like you, the participants you followed this round updated their choice based on their first guess and the revised answer of the participants they follow :\n" + neighbor_responses + "\n\n" + advised_choice_text
                agent.add_message_to_discussion(text)
            response_tasks = [agent.get_answer(tools=self._correlation_tool, tool_choice="auto") 
                            for agent in self.agents]
            responses = await asyncio.gather(*response_tasks)
            for message_content, choice, agent_id in responses:
                self.game[round][agent_id][GameStage.advised_choice].setdefault("choice", []).append(choice)
                self.game[round][agent_id][GameStage.advised_choice].setdefault("content", []).append(message_content)

    def give_results(self, round, correct_answer, advised_choice = False):
        result = lambda answer: f"This round({round+1}), your final choice was the value {answer}. The exact result was {correct_answer}\n"
        choice_stage = GameStage.advised_choice if advised_choice else GameStage.first_choice
        for agent in self.agents:       
            choice = self.game[round][agent.id][choice_stage]["choice"]
            answer = choice[-1] if advised_choice else choice
            self.game[round][agent.id]["score"] = abs(answer - correct_answer)
            text = result(answer)
            agent.add_message_to_discussion(text)

    async def run_neighbors_choice(
            self, 
            round, 
            ):
        last_answer = lambda agent_id, answer: f"- Participant {agent_id} last answer was {answer}"
        for agent in self.agents:
            neighbors_results = '\n'.join(
                [last_answer(neigbor_id, self.game[round][neigbor_id][GameStage.advised_choice]["choice"][-1]) for neigbor_id in agent.neigbors]
            )
            text = (
                "\nHere are the last answers of the participants you followed this round:\n"
                f"{neighbors_results}\n"
            )
            other_participant_results = '\n'.join(
                [
                 last_answer(other_agents.id, self.game[round][other_agents.id][GameStage.advised_choice]["choice"][-1]) 
                 for other_agents in self.agents 
                 if (other_agents.id not in agent.neigbors and agent.id != other_agents.id)]
            )
            text += (
                "\nHere are the answers of the participants you did not follow this round:\n"
                f"{other_participant_results}\n"
            ) if len(other_participant_results) > 0 else "\n- They are no other participants\n"
            text = text + "\n" + neighbors_choice_text + f"\n- You can choose up to {self.max_neighbor} participants.\n"
            print(text)
            agent.add_content_to_last_message(text)
        response_tasks = [agent.get_answer(tools=self._get_neigbors_tool(agent.id, self.max_neighbor), 
                                        tool_choice="auto") for agent in self.agents]
        responses = await asyncio.gather(*response_tasks)
        for message_content, choice, agent_id in responses:
            self.game[round][agent_id][GameStage.neighbors_choice]["choice"] = choice
            self.game[round][agent_id][GameStage.neighbors_choice]["content"] = message_content
            self.agents[agent_id].neigbors = choice

    async def run_experiment(
            self, 
            save_discussions=True, 
            save_result=True,
            shuffle_task=False, 
            advised_choice=True, 
            change_neighbors=True, 
            self_feedback=True,
            condition="test"
    ):
        """
        Run the experiment

        """

        print(f"self_feedback : {self_feedback}, advised_choice : {advised_choice}, change_neighbors : {change_neighbors}")
        neighbors =  "\n".join([f"      - {agent.id} -> { ','.join(map(str, agent.neigbors))}" for agent in self.agents])
        diffculties = "\n".join([f"     - {agent.id} -> {agent.difficulty}" for agent in self.agents])
        intro = ("Starting experiment :\n" \
                f"  - Number of agents : {self.nb_agents}\n" \
                f"  - Number of rounds : {self.nb_rounds}\n" \
                f"  - Each agent has a maximum of {self.max_neighbor} neighbors\n" \
                f"  - Participant neighbors are :\n{neighbors}\n" \
                f"  - Participants difficulty are :\n{diffculties}"
        )
        print(intro)
        is_last_round = lambda round: round == self.nb_rounds - 1
        tasks = self.tasks if not shuffle_task else random.shuffle(self.tasks)
        for i in range(self.nb_rounds):
            task = tasks[i]
            correctAnswer = task["correctAnswer"] * 100
            print("-------------------------------------------------")
            print(f"Round {i+1}")
            print(f"Task id {task['_id']}, correct answer {correctAnswer}")
            print("-------------------------------------------------")

            for agent in self.agents:
                self.game[i][agent.id]["difficulty"] = agent.difficulty
                self.game[i][agent.id]["neighbors"] = agent.neigbors
            
            await self.run_first_stage(task=task, round=i)

            # remove image from discussion
            for agent in self.agents:
                agent.discussion[agent.image_index]["content"].pop()
            if advised_choice:
                await self.run_advised_choice(round=i, times=3)
            if self_feedback:
                self.give_results(
                    round=i,
                    correct_answer=correctAnswer,
                    advised_choice=advised_choice
                )

            if not is_last_round(i):
                # makes no sense to change neighbors if you don't even now the correct answer
                if change_neighbors and self_feedback:
                    await self.run_neighbors_choice(
                        round=i
                    ) 
            # Shock
            if (
                self._shockRate > 0 and 
                (i+1) % int(round(self.nb_rounds / (self._shockRate * self.nb_rounds))) == 0
                # nb_round = 10, shockRate = 0.2, 10/2 = 5, 5 % 5 = 0
                # i = 4 --> round = 5 and we want the shock to happen after round 5                    
            ):
                self._shock(round=i+1)

            for agent, choice in self.game[i].items():
                print(f"Agent {agent} :")
                for stage in GameStage:
                    print(f"    - Stage {stage.name} : {choice[stage].get('choice')}")
                print(f"    - Neighbors : {','.join(map(str, choice['neighbors']))}")
                print(f"    - Difficulty : {choice['difficulty']}")
            self.game[i]["correctAnswer"] = round(correctAnswer)
        
        for agent in self.agents:
            # print cumulative score for each agent
            cumulative_score = sum([self.game[round][agent.id]["score"] for round in range(self.nb_rounds)])
            print(f"Agent {agent.id} cumulative score is {cumulative_score}")
            file_path = f"results/{condition}/{self._game_id}"
            if save_result:
                os.makedirs(file_path, exist_ok=True)
                self.save_result(file_path, condition)

            if save_discussions:
                os.makedirs(f"results/{self._game_id}", exist_ok=True)
                agent.save_discussion(f"{file_path}/{agent.id}.txt")

    def _shock(self, round):
        """ change difficulty for each agent """
        print("Shock time, round", round)
        for agent in self.agents:
            agent.difficulty = Difficulty(((Difficulty[agent.difficulty].value - 1) % 3)).name
            print(f"Agent {agent.id} new difficulty changed to {agent.difficulty}")


    def save_result(self, file_path, condition):
        """ save the result of the experiment as a csv file """
        data = []
        for agent in self.agents:
            for round in range(self.nb_rounds):
                data.append(
                    {
                        "player_id": agent.id,
                        "condition": condition,
                        "round_index": round,
                        "correct_answer": self.game[round]["correctAnswer"],
                        "independent_guess": self.game[round][agent.id][GameStage.first_choice]["choice"],
                        "revised_guess": self.game[round][agent.id][GameStage.advised_choice].get("choice",[None])[-1],
                        "neighbors_choice": self.game[round][agent.id][GameStage.neighbors_choice].get("choice"),
                        "score": self.game[round][agent.id]["score"],
                        "difficulty": self.game[round][agent.id]["difficulty"],
                        "alter_1": self.game[round][agent.id]["neighbors"][0] if len(self.game[round][agent.id]["neighbors"]) > 0 else None,
                        "alter_2": self.game[round][agent.id]["neighbors"][1] if len(self.game[round][agent.id]["neighbors"]) > 1 else None,
                        "alter_3": self.game[round][agent.id]["neighbors"][2] if len(self.game[round][agent.id]["neighbors"]) > 2 else None,
                        "in_degree": len(self.game[round][agent.id]["neighbors"]),
                    }
                )
        df = pd.DataFrame(data)

        # set independent_error for every row
        df["independent_error"] = abs(df["correct_answer"] - df["independent_guess"])
        df["revised_error"] = abs(df["correct_answer"] - df["revised_guess"])
        df["cumulative_score"] = df.groupby("player_id")["score"].cumsum()
        df["increment"] = df["cumulative_score"].diff()
        df["round_after_shock"] = df["round_index"].apply(lambda x: x % int(self.nb_rounds / self._shockRate) if self._shockRate > 0 else None)

        df.to_csv(f"{file_path}/data.csv", index=False)

    def print_discussion(self, agent_id):
        """ print text conversation """
        for message in self.agents[agent_id].discussion:
            print(message["role"], ":", message["content"][0]["text"])
            print()

    @staticmethod
    def generate_scatter_plot(corellation_value, nb_points , image_path):
        """ generate scatter plot with a given corellation along a random linear line and return it"""
        x = [random.random() for _ in range(nb_points)]
        y = [corellation_value * x[i] + random.random() for i in range(nb_points)]
        plt.scatter(x, y)
        plt.savefig(f"scatter_plt/{image_path}.png")
        plt.close()


    def generate_images(self, correlations):
        """ generate images for the experiment"""
        image_data = []
        for i in range(len(correlations)):
            correlation = random.random()
            index = i%10 + i//10 * 30
            image = {
                "_id": i,
                "correct_answer": correlation,
                "difficultyPath": {
                    "high": f"scatter_plt/image_{index}.png",
                    "medium": f"scatter_plt/image_{index+10}.png",
                    "easy": f"scatter_plt/image_{index+20}.png"
                }
            }
            image_data.append(image)
            ExperimentGroup.generate_scatter_plot(correlations[i], nb_points=20, image_path= f"image_{index}")
            ExperimentGroup.generate_scatter_plot(correlations[i], nb_points=40, image_path= f"image_{index + 10}")
            ExperimentGroup.generate_scatter_plot(correlations[i], nb_points=60, image_path= f"image_{index + 20}")
        return image_data
    
    
    def _set_neigbors_randomly(self):
        """ set max_neighbor amount randomly for each agent without including the agent itself """
        for agent in self.agents:
            agent.neigbors = [neighbor.id for neighbor in random.sample([a for a in self.agents if a.id != agent.id], self.max_neighbor)]
    @staticmethod
    def set_initial_network(num_agents, num_neighbors):
        """ set initial network for the agents """
        initial_network = {}
        for i in range(num_agents):
            initial_network[i] = random.sample([j for j in range(num_agents) if j != i], num_neighbors)
        return initial_network