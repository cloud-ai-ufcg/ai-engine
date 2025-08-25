import os
from threading import Lock
from openai import OpenAI


class OpenRouterClient:
    """
    Singleton client for OpenRouter API.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """
        Returns a singleton instance of the OpenRouterClient.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OpenRouterClient, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initializes the OpenRouterClient.
        """
        if self._initialized:
            return
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment variables")
        self.client = OpenAI(
            api_key=self.api_key, base_url="https://openrouter.ai/api/v1"
        )
        self._initialized = True

    def chat(self, model: str, system_prompt: str, user_prompt: str, **config):
        """
        Returns a chat completion for the given model, system prompt, and user prompt.

        Args:
            model (str): The model to use for the chat completion.
            system_prompt (str): The system prompt for the chat completion.
            user_prompt (str): The user prompt for the chat completion.
            **config: Additional configuration for the chat completion.

        Returns:
            str: The chat completion.
        """
        if not system_prompt or not user_prompt:
            raise ValueError("system_prompt and user_prompt are required")

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **config
        )
        return response.choices[0].message.content
