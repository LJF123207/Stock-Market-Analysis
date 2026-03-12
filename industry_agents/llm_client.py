from __future__ import annotations

import os
from typing import Any


class LLMClient:
    def __init__(self) -> None:
        self._model = self._build_model()

    def _build_model(self) -> Any:
        provider = os.getenv("LLM_PROVIDER", "openai-compatible").lower()
        model_name = os.getenv("LLM_MODEL", "glm-5-fp8")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

        api_key = os.getenv("OPENAI_API_KEY", "xxxxx").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "xxxxxxx").strip()

        if provider in {"openai", "openai-compatible"}:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency: langchain-openai. Install with `pip install langchain-openai`."
                ) from exc
            kwargs = {"model": model_name, "temperature": temperature}
            #api_key = os.getenv("OPENAI_API_KEY", "").strip()
            #base_url = os.getenv("OPENAI_BASE_URL", "").strip()
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            return ChatOpenAI(**kwargs)

        raise RuntimeError("Unsupported LLM_PROVIDER. Use `openai-compatible` or `openai`.")

    def ask(self, *, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self._model.invoke([SystemMessage(system), HumanMessage(user)])
        return response.content if isinstance(response.content, str) else str(response.content)


_LLM: LLMClient | None = None


def get_llm() -> LLMClient:
    global _LLM
    if _LLM is None:
        _LLM = LLMClient()
    return _LLM
