"""
Code Embeddings Module

This module provides functionality for generating embeddings with OpenAI, storing them in FAISS,
and retrieving similar code snippets based on semantic similarity. It also includes dependency
graph analysis for code projects.

Created by Datoscout at 18/02/2025
solutions@datoscout.ec
"""

# Standard library imports
from __future__ import annotations

import ast
import json
import os
import pickle

# Third-party library imports
import faiss
import networkx as nx
import numpy as np
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

# Local application imports
from proyecto_code.directed_graph import (
    analyze_project,
    filter_graph,
    visualize_directed_graph_interactive,
)
from proyecto_code.prompts.base import prompt_base
from proyecto_code.serialize import build_dependencies_files
from src.config.settings import (
    DATA_PATH,
    OPENAI_API_KEY,
    OPENAI_COMPLETIONS_MODEL,
    OPENAI_EMBEDDINGS_MODEL,
)
from src.proyecto_code.project_tree import main

# Create data directory if it doesn't exist
CODE_FOLDER = DATA_PATH / "code"
os.makedirs(CODE_FOLDER, exist_ok=True)


class Answer(BaseModel):
    """
    Response model for AI-generated answers.

    Attributes:
        response (str): The text response from the AI.
        code_snippet (str): Any code snippet included in the response.
    """

    response: str
    code_snippet: str


class CoderAI:
    """
    A class for analyzing code projects, generating embeddings, and retrieving similar code snippets.

    This class extracts code snippets from a project, generates embeddings using OpenAI,
    stores them in a FAISS index, and provides methods to search for similar code and
    analyze dependencies.

    Attributes:
        INDEX_NAME (str): Filename for the FAISS index.
        META_NAME (str): Filename for the metadata pickle file.
        DEPENDENCIES_GRAPH (str): Filename for the dependency graph.
        index_path (str): Full path to the FAISS index file.
        meta_path (str): Full path to the metadata file.
        graph_path (str): Full path to the dependency graph file.
        index: The FAISS index for similarity search.
        metadata: Metadata for code snippets.
        graph: The dependency graph for the project.
    """

    INDEX_NAME = "code_embeddings.index"
    META_NAME = "metadata.pkl"
    DEPENDENCIES_GRAPH = "dependencies_graph.gml"
    index_path = os.path.join(DATA_PATH, INDEX_NAME)
    meta_path = os.path.join(DATA_PATH, META_NAME)
    graph_path = os.path.join(DATA_PATH, DEPENDENCIES_GRAPH)
    index = None
    metadata = None
    graph = None

    def __init__(self, project_path: str):
        """
        Initialize the CoderAI with a project path.

        Args:
            project_path (str): The path to the code project to analyze.
        """
        self.project_path = project_path
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.set_data()

    def set_data(self) -> None:
        """
        Set up the data for the project.

        This method either generates embeddings, dependency graph, and metadata from scratch
        if they don't exist, or loads existing data from disk.

        Raises:
            Exception: If there is an error loading the metadata or index.
        """
        if not os.path.exists(self.index_path):
            # Step 1: Extract the dependency graph and dependencies from the project
            dependency_graph, dependencies = analyze_project(self.project_path)
            logger.info("Dependency extraction complete.")
            # Save the dependency graph for later use
            nx.write_gml(dependency_graph, os.path.join(CODE_FOLDER, self.DEPENDENCIES_GRAPH))

            # Step 2: Extract code snippets (functions and classes) using the project dependencies
            code_snippets, metadata = self.extract_code_snippets(dependencies)
            logger.info(f"Extracted {len(code_snippets)} code snippets from the project.")

            # Step 3: Compute embeddings for each code snippet
            embeddings = np.array([self.get_code_embedding(snippet) for snippet in code_snippets])
            logger.info("Computed embeddings for all code snippets.")

            # Step 4: Save embeddings and metadata to disk
            self.save_embeddings_and_metadata(embeddings, metadata)
            logger.info("Embeddings and metadata saved to disk.")

        try:
            # Load index from disk
            self.index = faiss.read_index(self.index_path)
            # Load graph from disk
            self.graph = nx.read_gml(self.graph_path)
            # Load metadata from disk
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)

        except Exception as e:
            logger.error(f"Error loading metadata: {e} -> delete all content in {CODE_FOLDER}")
            raise

    def get_code_embedding(self, code_snippet: str) -> np.ndarray:
        """
        Generate an embedding for a given code snippet using OpenAI.

        Args:
            code_snippet (str): The code snippet to embed.

        Returns:
            np.ndarray: A normalized 1D numpy array representing the embedding.
        """
        # Request embedding from OpenAI
        response = self.client.embeddings.create(
            input=code_snippet, model=OPENAI_EMBEDDINGS_MODEL, dimensions=1536
        )
        emb = [data.embedding for data in response.data]
        embedding = np.array(emb)

        # Normalize the embedding to unit vector for cosine similarity
        norms = np.linalg.norm(embedding)
        embedding = embedding / norms

        return embedding.astype("float32").squeeze()

    def generate_answer(self, prompt: str) -> Answer | None:
        """
        Generate an AI response to a prompt using OpenAI.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            Answer | None: An Answer object containing response and code_snippet if successful,
                          None if an error occurs.

        Note:
            Temperature=0 is used for deterministic outputs.
            Top_p=1 means no truncation of token selection.
            Frequency_penalty=0 means no penalty for repeating information.
            Presence_penalty=0 means the model is neutral about introducing new topics.
        """
        try:
            completion = self.client.beta.chat.completions.parse(
                model=OPENAI_COMPLETIONS_MODEL,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                response_format=Answer,
            )

            return completion.choices[0].message.parsed

        except KeyError as e:
            # Handle missing keys in the response
            logger.error(f"KeyError: {e} - The expected key is not in the response.")
            logger.info("An error occurred: the response structure was not as expected.")
            return None

        except Exception as e:
            # Handle any other general exceptions
            logger.error(f"An error occurred: {e}")
            logger.info("An error occurred while generating the response.")
            return None

    def extract_code_snippets(self, dependencies: dict) -> tuple[list[str], list[dict]]:
        """
        Extract code snippets from files based on dependency information.

        Uses AST to extract only the relevant code blocks (functions and classes)
        from the project files, based on the dependency information.

        Args:
            dependencies (dict): Dictionary mapping file paths to dependency data.

        Returns:
            tuple[list[str], list[dict]]: A tuple containing:
                - code_snippets: A list of code snippet strings.
                - metadata: A list of dicts with 'type', 'name', and 'file' keys.
        """
        code_snippets = []
        metadata = []

        def get_source_segment(file_path: str, node: ast.AST) -> str:
            """
            Extract the exact code block for a function/class from the source file.

            Args:
                file_path (str): Path to the source file.
                node (ast.AST): The AST node representing the function or class.

            Returns:
                str: The source code of the function or class.
            """
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
            return "".join(lines[node.lineno - 1 : node.end_lineno])

        for file, deps in dependencies.items():
            try:
                with open(file, encoding="utf-8") as f:
                    source_code = f.read()
            except Exception as e:
                logger.error(f"Error reading file {file}: {e}")
                continue

            try:
                # Parse the file into an AST
                tree = ast.parse(source_code, filename=file)
            except SyntaxError as e:
                logger.error(f"Error parsing file {file}: {e}")
                continue

            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in deps.get("functions", []):
                    snippet = get_source_segment(file, node)
                    code_snippets.append(snippet)
                    metadata.append({"type": "function", "name": node.name, "file": file})

                elif isinstance(node, ast.ClassDef) and node.name in deps.get("classes", []):
                    snippet = get_source_segment(file, node)
                    code_snippets.append(snippet)
                    metadata.append({"type": "class", "name": node.name, "file": file})

        return code_snippets, metadata

    def save_embeddings_and_metadata(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Save computed embeddings into a FAISS index and store metadata as a pickle file.

        Args:
            embeddings (np.ndarray): Array of embeddings with shape (num_snippets, embedding_dim).
            metadata (list[dict]): List of metadata dictionaries for each embedding.
        """
        # Determine the dimensionality of the embeddings
        d = embeddings.shape[1]

        # Create a FAISS index for Inner product (equivalent to cosine similarity for normalized vectors)
        index = faiss.IndexFlatIP(d)

        # Add embeddings to the FAISS index
        index.add(embeddings.astype("float32"))

        # Write the FAISS index to disk
        faiss.write_index(index, self.index_path)

        # Save metadata to disk using pickle
        with open(self.meta_path, "wb") as f:
            pickle.dump(metadata, f)

    def search_similar_code(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve similar code snippets for a given query using FAISS.

        Args:
            query (str): Query code snippet or description.
            top_k (int): Number of similar results to retrieve. Defaults to 5.

        Returns:
            list[dict]: A list of metadata dictionaries for the similar code snippets.
        """
        # Generate the embedding for the query
        query_embedding = self.get_code_embedding(query)

        # Perform the similarity search
        cos_simil, indices = self.index.search(np.array([query_embedding]), top_k)

        # Log distances and indices for debugging
        for dist, idx in zip(cos_simil[0], indices[0]):
            logger.debug(f"Index: {idx}, Similarity: {dist}")

        # Collect and return results using the indices
        results = [self.metadata[i] for i in indices[0]]
        return results

    def get_dependencies_subgraph(self, file_paths: list[str], level: int = 5) -> nx.Graph:
        """
        Retrieve a subgraph of dependencies for the given file paths.

        Args:
            file_paths (list[str]): List of file paths extracted from metadata.
            level (int): Maximum number of dependency levels (hops) to include. Defaults to 5.

        Returns:
            nx.Graph: A subgraph containing nodes reachable within 'level' hops from
                     any node associated with the input files.
        """
        selected_nodes = set()
        # Remove unknown nodes from the graph
        filtered_graph = filter_graph(self.graph, "remove", "unknown")

        # Find nodes that correspond to the input files
        for node in filtered_graph.nodes():
            for file in file_paths:
                if file in node:
                    selected_nodes.add(node)
                    break

        # For each selected node, get its ego graph up to the specified radius
        sub_nodes = set()
        for node in selected_nodes:
            ego = nx.ego_graph(filtered_graph, node, radius=level)
            sub_nodes.update(ego.nodes())

        # Create and return the subgraph
        subgraph = filtered_graph.subgraph(sub_nodes).copy()
        return subgraph


if __name__ == "__main__":
    # Example usage of the CoderAI class
    project_path = r"C:\Users\ecepeda\OneDrive - analitika.fr\Documentos\PROYECTOS\ANALITIKA\PycharmProjects\neural_networks\coder"

    coder = CoderAI(project_path)

    # Sample query to find relevant code snippets
    query_code = "where is the CoderAI class"
    similar_snippets = coder.search_similar_code(query_code)

    # Extract unique file paths from the returned metadata
    file_paths = list({snippet_meta["file"] for snippet_meta in similar_snippets})

    # Retrieve a dependency subgraph with dependencies up to 3 levels
    dep_subgraph = coder.get_dependencies_subgraph(file_paths, level=3)
    visualize_directed_graph_interactive(dep_subgraph, coder.project_path)

    # Print dependencies
    print("Dependencies:")
    dependencies, dep_files = [], []
    for u, v, data in dep_subgraph.edges(data=True):
        u_file = u.split("::")[0]
        v_file = v.split("::")[0]
        if u_file not in dep_files:
            dep_files.append(u_file)
        if v_file not in dep_files:
            dep_files.append(v_file)
        msg = f"{u.replace(project_path, '.')} -> {v.replace(project_path, '.')}: {data['type']}"
        if u not in dependencies:
            dependencies.append(msg)

    # Build file content and project structure
    file_content = build_dependencies_files(dep_files)
    project_files = json.dumps(file_content)
    project_structure = main(project_path, only_py=True, write_file=False)

    # Example project description for generating documentation
    project_description = """
        This is a project develops a RAG for helping developers to improve coding in a
        more effective and faster way. We use OpenAI to compute embeddings and retrieve the
        snippets and files more relevant to the query. The project is developed in Python.
        """

    # Example task for the AI to perform
    task_question = """
        Make the documentation for the CoderAI class. Consider all the given information.
    """

    # Format prompt and generate an answer
    prompt = prompt_base.format(
        project_name="Code IA-RAG",
        project_description=project_description,
        project_files=project_files,
        dependencies_files=dependencies,
        project_structure=project_structure,
        task_question=task_question,
    )

    ans_ = coder.generate_answer(prompt)
    print(ans_.response)
    print(ans_.code_snippet)
