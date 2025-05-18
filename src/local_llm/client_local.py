# External imports
import torch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Chat:
    """
    Class to simulate the chat attribute of OpenAI client.
    """

    def __init__(self, parent: "LocalAI") -> None:
        self.parent = parent
        self.completions = ChatCompletions(parent)


class ChatCompletions:
    """
    Class to simulate the OpenAI chat completions structure.
    """

    def __init__(self, parent: "LocalAI") -> None:
        self.parent = parent

    def create(self, messages: list = None, **kwargs):
        """
        Create a chat completion similar to OpenAI's interface.

        Args:
            messages (list): List of message dictionaries with role and content
            **kwargs: Additional arguments (ignored for simplicity)

        Returns:
            ChatResponse: A response object containing the generated text
        """
        # Convert OpenAI-style messages to LangChain message format
        langchain_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        # Generate response using LangChain
        response = self.parent.chat_model.invoke(langchain_messages)

        # Format response to match OpenAI structure
        return ChatResponse(response.content)


class ChatResponse:
    """
    A class to simulate OpenAI's response structure.
    """

    def __init__(self, content: str) -> None:
        self.choices = [Choice(content)]


class Choice:
    """
    A class to simulate OpenAI's choice structure.
    """

    def __init__(self, content: str) -> None:
        self.message = Message(content)


class Message:
    """
    A class to simulate OpenAI's message structure.
    """

    def __init__(self, content: str) -> None:
        self.content = content


class LocalAI:
    """
    A class that implements a local LLM client interface similar to OpenAI's client.
    This class wraps around a LangChain pipeline to provide a similar interface
    to the OpenAI client but for local models.
    """

    def __init__(
        self, model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct", by_api: bool = False
    ) -> None:
        """
        Initialize the LocalAI client with a specified model.
        - model_id="microsoft/Phi-3-mini-4k-instruct"
        - model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct"
        Args:
            model_id (str): The Hugging Face model ID to use
            by_api (bool): Whether to use the API or the local model
        """
        self.model_id = model_id
        self.by_api = by_api
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Print CUDA information if available
        if self.device == "cuda":
            print("CUDA available:", torch.cuda.is_available())
            print("CUDA device count:", torch.cuda.device_count())
            print("CUDA device name:", torch.cuda.get_device_name(0))
            print("CUDA compute capability:", torch.cuda.get_device_capability(0))
            print(
                "Memory total (MB):", torch.cuda.get_device_properties(0).total_memory / 1024**2
            )

        # Initialize the model and tokenizer
        if not self.by_api:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, truncation=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )

        # Initialize pipeline and LangChain components
        self._init_pipeline()

        # Initialize the chat attribute to match OpenAI client structure
        self.chat = Chat(self)

    def _init_pipeline(self) -> None:
        """Initialize the text generation pipeline and LangChain components."""

        if self.by_api:
            self.llm = HuggingFaceEndpoint(
                repo_id=self.model_id,
                task="text-generation",
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                return_full_text=False,
                repetition_penalty=1.03,
            )
        else:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,  # when temperature used, do_sample is True
                return_full_text=False,
                repetition_penalty=1.03,
                # presence_penalty=1.03,
            )

            # Create LangChain HuggingFacePipeline
            self.llm = HuggingFacePipeline(pipeline=self.pipeline)

        # Create ChatHuggingFace model
        self.chat_model = ChatHuggingFace(llm=self.llm)
