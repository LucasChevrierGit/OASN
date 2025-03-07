import openai

# Define the function the model can call
tools = [
    {
        "type": "function",
        "function": {
            {
                "name": "choose_multiple_integers",
                "description": "Select three distinct integers from the given options.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_values": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "enum": [1, 2, 3, 4, 5]
                            },
                            "description": "An array of three distinct integers.",
                            "minItems": 3,
                            "maxItems": 3,
                            "uniqueItems": True
                        }
                    },
                    "required": ["selected_values"]
                }
            }
        }
    }
]

functions = [

]
# Initialize the OpenAI client
client = openai.OpenAI()

# Make the API call with tool_choice set to "required"
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Please choose an integer."}
    ],
    tools=tools,
    tool_choice="required"
)



print(response.choices[0].message)
