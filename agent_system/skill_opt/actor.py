"""Frozen Actor = OpenAI-compatible client against a local vllm server."""
import concurrent.futures as cf
from openai import OpenAI


class ActorClient:
    def __init__(self, base_url="http://127.0.0.1:8100/v1", model="actor",
                 api_key="EMPTY", temperature=0.7, top_p=0.8, top_k=20,
                 repetition_penalty=1.05, max_tokens=512,
                 max_workers=48, timeout=180):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=3)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.max_workers = max_workers

    def _one(self, prompt, temperature):
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                # top_k + repetition_penalty are vllm extensions (not native OpenAI params)
                extra_body={"top_k": self.top_k, "repetition_penalty": self.repetition_penalty},
            )
            return r.choices[0].message.content or ""
        except Exception:
            return ""  # invalid output -> env treats it as an invalid action

    def generate(self, prompts, temperature=None):
        """Batched chat completions for a list of prompts (order preserved)."""
        t = self.temperature if temperature is None else temperature
        out = [""] * len(prompts)
        if not prompts:
            return out
        with cf.ThreadPoolExecutor(max_workers=min(self.max_workers, len(prompts))) as ex:
            futs = {ex.submit(self._one, p, t): i for i, p in enumerate(prompts)}
            for f in cf.as_completed(futs):
                out[futs[f]] = f.result()
        return out

    def healthy(self):
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
