import os
import base64
from openai import AzureOpenAI
from dotenv import load_dotenv
import numpy as np
import time

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        endpoint = os.getenv("gpt_endpoint")
        api_key = os.getenv("gpt_api_key")
        api_version = os.getenv("gpt_api_version")
        _client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    return _client


def generate_response(text_input):
    """
    Input:
        text_input (str): The user's chat input.

    Output:
        response (str): A Markdown-formatted response from the agent.
    """
    start_time = time.time()
    deployment = os.getenv("gpt_deployment")
    client = _get_client()

    # Prepare the full chat prompt with system and user messages
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """You are a helpful assistant working for Zava, a company that specializes in offering products to assist homeowners with do-it-yourself projects.
                        Respond to customer inquiries with relevant product recommendations and DIY tips. If a customer asks for paint, suggest one of the following three colors: blue, green, and white.
                        If a customer asks for something not related to a DIY project, politely inform them that you can only assist with DIY-related inquiries.
                        Zava has a variety of store locations across the country. If a customer asks about store availability, direct the customer to the Miami store.
                    """,
                }
            ],
        },
        {"role": "user", "content": text_input},
    ]

    # Call Azure OpenAI chat API
    completion = client.chat.completions.create(
        model=deployment,
        messages=chat_prompt,
        max_completion_tokens=10000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
    )
    end_sum = time.time()
    print(f"generate_response Execution Time: {end_sum - start_time} seconds")
    # Return response content
    return completion.choices[0].message.content
