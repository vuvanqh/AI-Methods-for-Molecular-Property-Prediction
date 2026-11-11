import ssl

import httpx

from .core.config import settings


class LLMClient:
    def __init__(self, base_url: str = settings.llm_service_url):
        self.url = base_url

    async def get_summary(self, user_prompt: str):
        """https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion"""

        try:
            async with httpx.AsyncClient(
                timeout=settings.llm_timeout_seconds,
                # the system trust store, since certifi misses the root re-signing HTTPS here
                verify=ssl.create_default_context(),
            ) as client:
                response_body = {
                    "max_tokens": settings.llm_max_tokens,
                    "messages": [
                        {
                            "content": "You are a chemistry expert with excellent summarization skills.",
                            "role": "system",
                        },
                        {"content": user_prompt, "role": "user"},
                    ],
                    "model": settings.llm_model,
                    "temperature": settings.llm_temp,
                }
                headers = {"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"}

                response = await client.post(
                    self.url, json=response_body, headers=headers
                )
                response.raise_for_status()
        except httpx.RequestError as exc:
            raise ConnectionError("LLM service unavailable") from exc

        except httpx.HTTPStatusError as exc:
            raise ConnectionError(f"LLM service returned {exc.response.status_code}")

        try:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Obtained unexpected response format") from exc

        if not answer:
            # reasoning models can spend the whole token budget before writing any content
            raise RuntimeError("LLM returned an empty summary")

        return answer


llm_client = LLMClient()
