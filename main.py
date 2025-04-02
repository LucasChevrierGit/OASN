from openai import OpenAI
from experimentGroup import ExperimentGroup
import asyncio

with open('prompts/basic_preprompt.txt', 'r') as file:
    preprompt = file.read()

initial_network = {
    0: [2,5],
    1: [0,2],
    2: [3,5],
    3: [4,0],
    4: [1,2],
    5: [4,3],
}

experiments_params = {
    "dynamic_network_2": {"advised_choice": True, "change_neighbors": True, "self_feedback": True},
    "static_network_2": {"advised_choice": True, "change_neighbors": False, "self_feedback": True},
}

for condition, params in experiments_params.items():
    print(f"Running experiment for condition: {condition}")
    for _ in range(3):
        experiment = ExperimentGroup(nb_rounds= 10, 
                            nb_agents= 6, 
                            pre_prompts= [preprompt] * 6,
                            max_neighbor=2, 
                            initial_network = initial_network,
                            shockRate=0.2,
        )
        asyncio.run(experiment.run_experiment(**params, condition=condition))

