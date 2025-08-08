# External imports
import time

import streamlit as st
from openai import OpenAI

# Internal imports
from src.config.settings import OPENAI_API_KEY, OPENAI_ASSISTANT_ID

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")


def create_thread():
    """Create a new thread for the assistant conversation"""
    return client.beta.threads.create()


def submit_message(thread, user_message: str):
    """Submit a message to the assistant thread"""
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_message)
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=OPENAI_ASSISTANT_ID,
    )


def get_response(thread) -> list:
    """Get messages from the thread"""
    return client.beta.threads.messages.list(thread_id=thread.id, order="desc")


def wait_on_run(run, thread):
    """Wait for the assistant to complete its response"""
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.1)
    return run


def decode_assistant_response(messages: list) -> str:
    """Extract the assistant's response from messages"""
    response = ""
    for m in messages:
        if m.role == "assistant":
            response = m.content[0].text.value
            break
    return response


def get_thread_messages(thread) -> list:
    """Get all messages from the thread formatted for display"""
    messages = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    formatted_messages = []

    for message in messages.data:
        formatted_messages.append({"role": message.role, "content": message.content[0].text.value})

    return formatted_messages


def main() -> None:
    st.title("🤖 Asistente IA - Webapp")

    # Initialize session state
    if "thread" not in st.session_state:
        st.session_state.thread = create_thread()
        st.session_state.messages = []

    # Display chat history
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        # Display welcome message
        with st.chat_message("assistant"):
            st.markdown("¡Hola! Soy tu asistente IA. ¿En qué puedo ayudarte hoy?")

    # Chat input
    prompt = st.chat_input("¿Qué te gustaría saber?")
    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                # Submit message to assistant
                run = submit_message(st.session_state.thread, prompt)

                # Wait for response
                run = wait_on_run(run, st.session_state.thread)

                # Get and display response
                assistant_messages = get_response(st.session_state.thread)
                bot_response = decode_assistant_response(assistant_messages)

                st.markdown(bot_response)

                # Add assistant response to session state
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Sidebar with additional options
    with st.sidebar:
        st.header("Opciones")

        if st.button("🗑️ Limpiar Chat"):
            st.session_state.thread = create_thread()
            st.session_state.messages = []
            st.rerun()

        if st.button("📋 Mostrar ID del Thread"):
            st.info(f"Thread ID: {st.session_state.thread.id}")

        # Display conversation statistics
        if st.session_state.messages:
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_messages = len(
                [m for m in st.session_state.messages if m["role"] == "assistant"]
            )

            st.metric("Mensajes del Usuario", user_messages)
            st.metric("Respuestas del Asistente", assistant_messages)


if __name__ == "__main__":
    # streamlit run src/chatbots/d_asistente_webapp.py
    main()
