"""
Created by Claude 3.7 Sonnet at 18/02/2025

This script implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain,
but adapted to use our project's dependency graph-derived code embeddings (stored in a FAISS index)
and associated metadata. Instead of re-generating the vector base, the chatbot loads the FAISS index
from a given path and uses it as a retriever to supply ChatGPT with context from our project code.
"""

# External imports
import os
import sys

from langchain_community.vectorstores import FAISS  # Assumes our FAISS wrapper supports load_local
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

# Internal imports / configuration (update these paths/variables as needed)
from config import OPENAI_API_KEY, OPENAI_COMPLETIONS_MODEL, OPENAI_EMBEDDING_MODEL, PROJ_ROOT

# Path to our pre-generated FAISS index (e.g., "code_embeddings.index")
VECTOR_DB_PATH = os.path.join(PROJ_ROOT, "coder", "data")
INDEX_NAME = "code_embeddings.index.pkl"


class RAGChatbot:
    def __init__(self, vector_db_path: str):
        """
        Initialize the RAG chatbot using OpenAI embeddings, a pre-built FAISS vector store
        (containing our project's code elements), and a structured prompt.
        """
        logger.info("Initializing RAG Chatbot with project code embeddings")

        try:
            self.embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY
            )
            self.llm = ChatOpenAI(model=OPENAI_COMPLETIONS_MODEL, openai_api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize LLM or embeddings: {e}")
            sys.exit(1)

        # Load the pre-generated FAISS vector store from the provided path.
        self.vectorstore = self.create_vectorstore(vector_db_path)
        self.retriever = self.configure_retriever()
        self.prompt_template = self.create_prompt_template()

    def create_vectorstore(self, vector_db_path: str):
        """
        Load the FAISS vector store containing our project code embeddings and metadata.
        We assume that the FAISS index was generated and stored previously.
        """
        logger.info(f"Loading FAISS vector store from {vector_db_path}")
        try:
            # Load the vector store using LangChain's FAISS loader.
            # This method expects the index to be stored locally along with document metadata.
            vectorstore = FAISS.load_local(
                vector_db_path,
                self.embeddings,
                index_name=INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {e}")
            raise
        return vectorstore

    def configure_retriever(self):
        """
        Configure the retriever to perform similarity search on the code embeddings.
        The retriever is tuned to fetch a broader set of documents and then re-rank them.
        """
        retriever = self.vectorstore.as_retriever()
        retriever.search_kwargs.update(
            {
                "fetch_k": 20,  # Retrieve 20 documents before re-ranking.
                "maximal_marginal_relevance": True,  # Ensure diverse results.
                "k": 10,  # Return top 10 most relevant documents.
            }
        )
        logger.info("Retriever configured with search parameters: %s", retriever.search_kwargs)
        return retriever

    def create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create a structured prompt template that uses the retrieved code context.
        The context will consist of all relevant project code elements, extracted from our FAISS index.
        """
        logger.info("Creating structured prompt template")
        template = (
            "Answer the following question using ONLY the code context provided below.\n\n"
            "----- Code Context -----\n"
            "{context}\n"
            "----- End of Context -----\n\n"
            "Question: {question}\n\n"
            "Answer in a comprehensive and detailed manner, referencing the code if necessary."
        )
        try:
            prompt = ChatPromptTemplate.from_template(template)
        except Exception as e:
            logger.error(f"Error creating prompt template: {e}")
            raise
        return prompt

    def generate_answer(self, question: str) -> str:
        """
        Retrieve relevant code context from the vector store using the retriever,
        format the prompt, and generate an answer using the language model.
        """
        logger.info(f"Processing question: {question}")
        try:
            # Retrieve relevant documents (code snippets) from the FAISS vector store.
            # Here we assume that each document has a 'page_content' attribute containing the code.
            docs = self.retriever.get_relevant_documents(question)
            if not docs:
                logger.warning("No relevant code snippets found for the query.")
                context = "No code context available."
            else:
                # Concatenate the page contents of all retrieved documents.
                context = "\n".join([doc.page_content for doc in docs])
            logger.debug("Retrieved context: {}", context)

            # Prepare the prompt with the retrieved context and the question.
            prompt_input = {"context": context, "question": question}
            formatted_prompt = self.prompt_template.format(**prompt_input)
            logger.debug("Formatted prompt: {}", formatted_prompt)

            # Call the language model with the formatted prompt.
            raw_response = self.llm.invoke(formatted_prompt)
            logger.debug("LLM raw response: {}", raw_response)

            # Parse the raw response into a clean answer string.
            answer = StrOutputParser().parse(raw_response)
        except Exception as e:
            logger.error(f"Error during answer generation: {e}")
            answer = "Sorry, an error occurred while generating the answer."
        return answer


def main():
    logger.info("Starting RAG Chatbot application for project code retrieval")
    # If a vector store path is provided as a command-line argument, use it.
    vector_db_path = sys.argv[1] if len(sys.argv) > 1 else VECTOR_DB_PATH

    chatbot = RAGChatbot(vector_db_path)

    # Example questions to test the chatbot.
    # These questions should relate to your project's code elements.
    questions = [
        "How is the dependency graph constructed in this project?",
        "What function is responsible for extracting code embeddings?",
        "How does the RAG system integrate with ChatGPT?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        answer = chatbot.generate_answer(question)
        print(f"A{i}: {answer}")


if __name__ == "__main__":
    main()
