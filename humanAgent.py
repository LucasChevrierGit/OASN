
import json
from openai import OpenAI
import base64

CONTEXT_WINDOW = 128000 # gpt-4o-mini context window


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class HumanAgent:
    def __init__(self, client : OpenAI, id, preprompt, difficulty):
        self.id = id
        self.client = client
        self.system_message = preprompt
        self.discussion = [ 
            {
                "role": "system",
                "content": [{ "type": "text", "text": preprompt}]
            }
        ]
        self.total_tokens = 0

        self.cumulative_score = 0
        self.neigbors = []
        self.difficulty = difficulty

        self.image_index = None

    async def get_answer(self, temp=0.7, model="gpt-4o-mini", tools = None, tool_choice="auto"):
        if self.discussion[-1]["role"] != "user":
            raise("The last message should be from the user")
        
        if tool_choice=="required" and tools is None:
            raise ValueError("tools must be provided if tool_choice is required")
    
        completion = self.client.chat.completions.create(
            model=model,
            messages= self.discussion,
            temperature=temp,
            tools=tools,
            tool_choice=tool_choice
        )
        response = completion.choices[0].message
        argument_name = tools[0]["function"]["parameters"]["required"][0] if tools else None
        message_content = response.content
        self.discussion.append(
            {
                "role": "assistant",
                "content": [{ "type": "text", "text": message_content}],
                "tool_calls": response.tool_calls
            }
        )

        selected_value = None
        # If the user does not provide a value, we ask them to provide one
        if tools and response.tool_calls is None:
            print(f"---------------- ERROR : Agent {self.id} ----------------")
            self.add_message_to_discussion(
                "You must provide a value based on your previous message answer. Next time, please provide a value using the tool along with your message.")
            completion = self.client.chat.completions.create(
                model=model,
                messages= self.discussion,
                temperature=temp,
                tools=tools,
                tool_choice="required"
            )
            response = completion.choices[0].message
            self.discussion.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": response.tool_calls
                }
            )


        if response.tool_calls:
            for tool in response.tool_calls: 
                selected_value = json.loads(tool.function.arguments)[argument_name]
                self.discussion.append({
                    "role": "tool",
                    "content": f"The selected value is {selected_value}.",
                    "tool_call_id": tool.id
                }) 
        return message_content, selected_value, self.id
    
    def add_image_to_discussion(self, image_path: str, text: str):
        base64_image = encode_image(image_path)   
        self.discussion.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        )
        self.image_index = len(self.discussion) - 1
        token = self._count_message_tokens(self.discussion[-1])
        self.total_tokens += token

    def save_discussion(self, filename):
        text = ""
        for elem in self.discussion:
            if elem.get("content") is not None and elem['role'] != "tool":
                text += "\n".join([f"\n-------{elem['role']}--------\n{elem['content'][0]['text']}" ])
        with open(filename, "w") as file:
            file.write(text)

    def add_message_to_discussion(self, text: str):
        self.discussion.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": text
                    }
                ],
            }
        )

        token = self._count_message_tokens(self.discussion[-1])
        self.total_tokens += token

    def add_content_to_last_message(self, text: str):
        self.discussion[-1]["content"][0]["text"] += "/n" + text

    def _count_message_tokens(self, message):
        """Estimate token usage of a single message entry."""
        tokens = 0
        content = message.get("content", [])
        if not content:
            return tokens
        # Content may be a list of parts (text, image, etc.)
        for part in content:
            if part.get("type") == "text":
                text = part.get("text", "")
                tokens += len(text.split())                 # approximate by word count
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                tokens += len(url) // 4                      # rough estimate for base64 image data
        return tokens
