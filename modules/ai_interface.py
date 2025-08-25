import os
import time
import random
import logging
from typing import Callable, Dict, Optional, List, Any

try:
    from openai import OpenAI  # >= 1.0 client
    import openai
except Exception:
    OpenAI = None
    openai = None


class AIInterface:
    """Manage multiple AI providers (local llama.cpp, OpenAI, Baseten).
    Handles provider switching, quotas, and automatic fallback.
    """

    def __init__(
        self,
        key_loader,
        on_provider_changed: Optional[Callable[[str], None]] = None,
        on_capability_changed: Optional[Callable[[bool], None]] = None,
    ):
        self.key_loader = key_loader
        self.on_provider_changed = on_provider_changed
        self.on_capability_changed = on_capability_changed

        self.provider = "local"  # default
        self.client = None
        self.max_failures = 5
        self.failure_count = 0
        self.last_cloud_check = 0
        self.cloud_cap_exceeded = False
        self._init_provider(self.provider)

    # -------------------------------
    # Provider Management
    # -------------------------------

    def _init_provider(self, provider: str):
        """Initialize client for a given provider."""
        self.provider = provider
        api_key = self.key_loader.resolve_api_key(provider)

        if provider == "local":
            # local llama.cpp runs without API key
            self.client = None
            self.base_url = "http://127.0.0.1:8081/v1"
            self.model = "local"
        elif provider == "openai":
            if not api_key or OpenAI is None:
                self._disable_ask("No OpenAI key or library missing")
                return
            self.client = OpenAI(api_key=api_key)
            self.base_url = None
            self.model = "gpt-4o-mini"
        elif provider == "baseten":
            if not api_key:
                self._disable_ask("No Baseten key configured")
                return
            # Baseten proxies OpenAI-like API
            self.client = OpenAI(api_key=api_key, base_url="https://api.baseten.co/v1")
            self.base_url = "https://api.baseten.co/v1"
            self.model = "gpt-4o-mini"
        else:
            self._disable_ask(f"Unknown provider {provider}")
            return

        self._enable_ask()
        if self.on_provider_changed:
            self.on_provider_changed(provider)

    # -------------------------------
    # Public API
    # -------------------------------

    def ask(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """Submit a prompt to the current provider and return reply text.

        This method wraps both local llama.cpp and OpenAI/other providers.
        It handles quota, rate limits, fallbacks, and capability toggling.
        """
        if self.cloud_cap_exceeded and self.provider != "local":
            self._auto_fallback_to_local("Cloud cap exceeded")
            return "⚠️ Cloud cap exceeded, switched to local."

        try:
            if self.provider == "local":
                return self._ask_local(messages, max_tokens, temperature)
            elif self.provider in ("openai", "baseten"):
                return self._ask_openai(messages, max_tokens, temperature)
            else:
                return f"Provider {self.provider} not supported."
        except Exception as e:
            logging.exception("ask() failed")
            self.failure_count += 1
            if self.failure_count >= self.max_failures:
                self._auto_fallback_to_local(str(e))
                return f"⚠️ Error with {self.provider}, switched to local."
            return f"⚠️ {self.provider} error: {e}"

    # -------------------------------
    # Internal helpers
    # -------------------------------

    def _ask_local(self, messages, max_tokens, temperature):
        # TODO: implement local llama.cpp query (HTTP POST to self.base_url)
        return "[local inference not yet implemented]"

    def _ask_openai(self, messages, max_tokens, temperature):
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def _disable_ask(self, reason: str):
        if self.on_capability_changed:
            self.on_capability_changed(False)
        logging.warning("ASK disabled: %s", reason)

    def _enable_ask(self):
        if self.on_capability_changed:
            self.on_capability_changed(True)

    def _auto_fallback_to_local(self, reason: str):
        logging.warning("Falling back to local: %s", reason)
        if self.provider != "local":
            self._init_provider("local")
        else:
            # already local and failing, just re-enable ASK
            self._enable_ask()
