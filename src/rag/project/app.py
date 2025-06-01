"""
Created by Datoscout at 26/03/2024
solutions@datoscout.ec
"""
import random
import time

# streamlit run .\src\rag\b_basica\app.py
import boto3
# External imports
import streamlit as st

bedrockClient = boto3.client('bedrock-agent-runtime', 'us-east-1')


def chat_pdf(text: str) -> dict:
    return bedrockClient.retrieve_and_generate(
        input={'text': text},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                'knowledgeBaseId': 'WPI4TVOX3A',
                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": 5
                    }
                },
                "generationConfiguration": {
                    "promptTemplate": {
                        "textPromptTemplate": "You are a question answering agent. I will provide you with a set of search results. The user will provide you with a question. Your job is to answer the user's question using only information from the search results. If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question. \nJust because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion.\n\nHere are the search results in numbered order:\n$search_results$\n\n$output_format_instructions$\n\nResponse in spanish"
                    },
                    "inferenceConfig": {
                        "textInferenceConfig": {
                            "temperature": 0,
                            "topP": 1,
                            "maxTokens": 2048,
                            "stopSequences": [
                                "\nObservation"
                            ]
                        }
                    }
                },
                "orchestrationConfiguration": {

                }
            }
        },
    )


# Streamed response emulator
def response_generator(question: str):
    text = 'No comprendo. Consulta utilizando palabras diferentes.'
    response = chat_pdf(text=question)
    if isinstance(response, dict):
        output = response.get('output')
        if isinstance(output, dict):
            text = output.get('text')

    for word in text.split():
        yield word + " "
        time.sleep(0.05)


def title_generator():
    return random.choice(
        [
            "¡Hola! ¿En qué puedo ayudarte?",
            "Hola, ¿puedo ayudarte en algo?",
            "¿Necesitas ayuda?",
        ]
    )


def main():
    st.title("Proyecto ChatBot")
    st.markdown(
        """
Proyecto que busca medir los conocimientos adquiridos en el Curso **Desarrollo sofisticado de Chatbot**


## Estructura del Chatbot

El chatbot ha sido desplegado utilizando la infraestructura de AWS de la siguiente manera.

* IAM.- Habilitación de roles y permisos
* Amazon S3 (Repositorio de archivos)
* Creación de la base de conocimientos (knowledge-base)
* Uso de modelos (knowledge-base)
    * Anthropic > Claude 3 Haiku
* Key Management Service (KMS)


## Importante
No es necesario ingresar ninguna clave de acceso, solo es necesario realizar consultas tomando en cuenta el contenido del [Manual-Usuario-FirmaEC-v3.0.0.pdf](https://www.firmadigital.gob.ec/wp-content/uploads/2023/01/Manual-Usuario-FirmaEC-v3.0.0.pdf).        """,
        unsafe_allow_html=True,
    )

    st.title(title_generator())

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("¿Que es FirmaEc?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(question=prompt))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
