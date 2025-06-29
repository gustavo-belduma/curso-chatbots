# Standard imports

# Third party imports
from openai import OpenAI

from src.config.parameters import EXIT_WORDS

# Internal imports
import sys

from src.config.settings import OPENAI_API_KEY, OPENAI_COMPLETIONS_MODEL

# Set your OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")


def initial_message(welcome_message):
    """
    Create the initial message history.

    Args:
        welcome_message (str): The welcome message from the bot

    Returns:
        list: The initial message history
    """
    system_message = "Eres un asistente muy útil. Y hablas español."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": welcome_message},
    ]
    return messages


def format_response(response):
    """
    Format the response from the model.

    Args:
        response (ChatResponse): The response from the model

    Returns:
        str: The formatted response
    """
    return response.choices[0].message.content


def generate_response(message_history):
    """
    Generate a response from the model.

    Args:
        client (OpenAI): The OpenAI client
        message_history (list): The message history

    Returns:
        ChatResponse: The response from the model
    """

    response = client.chat.completions.create(
        model=OPENAI_COMPLETIONS_MODEL,  # Models: https://platform.openai.com/docs/models/overview
        messages=message_history,
        temperature=1.0,  # The temperature can range from 0 to 2.
        max_tokens=100,  # short answers
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response


def main() -> None:
    welcome_message = (
        "¡Hola! Soy tu chatbot. Pregúntame lo que quieras y haré lo mejor para ayudarte."
    )
    print(f"🤖: {welcome_message}")

    message_history = initial_message(welcome_message)

    while True:
        # Get user input
        print("🧑: ", end="")
        user_input = input()
        message_history.append({"role": "user", "content": user_input})

        # Check if the conversation is complete
        if any(exit_keyword in user_input.lower() for exit_keyword in EXIT_WORDS):
            print("🤖: ¡Chao! Sesión de chat finalizada.")
            break

        # Generate and display the bot's response
        response = generate_response(message_history)
        bot_response = format_response(response)
        print(f"🤖: {bot_response}")

        # Add the bot's response to the chat history
        message_history.append({"role": "assistant", "content": bot_response})


if __name__ == "__main__":
    main()
