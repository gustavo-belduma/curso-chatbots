# External imports
import streamlit as st
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

# Internal imports
from src.config.settings import (  # DEEPSEEK_API_KEY,; DEEPSEEK_COMPLETIONS_MODEL,
    OPENAI_API_KEY,
    OPENAI_COMPLETIONS_MODEL,
)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")
# client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def initial_message(welcome_message: str) -> list:
    system_message = "Eres un asistente útil y amigable que responde en español."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": welcome_message},
    ]
    return messages


def format_response(response: ChatCompletion) -> str:
    return response.choices[0].message.content


def generate_response(message_history: list) -> ChatCompletion:
    response = client.chat.completions.create(
        model=OPENAI_COMPLETIONS_MODEL,
        messages=message_history,
        temperature=0.7,
        top_p=1.0,
        # top_k=40,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response


def main() -> None:
    st.title("🤖 Chatbot IA")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        welcome_message = "¡Hola! Soy tu chatbot. ¿En qué puedo ayudarte hoy?"
        st.session_state.messages = initial_message(welcome_message)

    # Display chat history
    for message in st.session_state.messages[1:]:  # Skip the system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("¿Qué te gustaría saber?")
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = generate_response(st.session_state.messages)
            print(type(response))
            bot_response = format_response(response)
            st.markdown(bot_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})


if __name__ == "__main__":
    # streamlit run src\chatbots\c_streamlit.py
    main()
