"""Local LLM Engine with GPU support"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from config.settings import DEVICE, LLM_MODEL, EMBED_MODEL

_engine = None


class LLMEngine:
    def __init__(self, llm_model: str = LLM_MODEL, embed_model: str = EMBED_MODEL):
        self.device = DEVICE
        print(f"Device: {self.device}")

        # LLM
        print(f"Loading LLM: {llm_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

        # Embeddings
        print(f"Loading Embeddings: {embed_model}")
        self.embedder = SentenceTransformer(embed_model, device=self.device)

        print("Engine ready")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        result = self.generator(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        return result[0]["generated_text"].strip()

    def embed(self, text: str) -> list:
        return self.embedder.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: list) -> list:
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def clean_ocr(self, text: str) -> str:
        text = self._fix_hyphenation(text)
        prompt = f"Düzelt ve temizle:\n{text}\n\nDüzeltilmiş:"

        result = self.generate(prompt, max_tokens=len(text) + 100)
        result = self._remove_artifacts(result)

        if len(result) < len(text) * 0.3:
            return text
        return result

    def extract_units(self, text: str, patterns: list) -> list:
        regex = re.compile('|'.join(patterns), re.IGNORECASE)
        if not regex.search(text):
            return []

        prompt = f"Askeri birlikleri listele:\n{text}\n\nBirlikler:"
        result = self.generate(prompt, max_tokens=200)

        units = []
        for line in result.split('\n'):
            line = line.strip()
            if len(line) > 50:
                continue
            match = regex.search(line)
            if match:
                units.append(match.group())

        return list(dict.fromkeys(units)) if units else [m.group() for m in regex.finditer(text)]

    def _fix_hyphenation(self, text: str) -> str:
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        return ' '.join(text.split())

    def _remove_artifacts(self, text: str) -> str:
        patterns = [
            r'^[*\s]*İşte düzeltilmiş.*?[:\s]*',
            r'^[*\s]*Düzeltilmiş.*?[:\s]*',
        ]
        for p in patterns:
            text = re.sub(p, '', text, flags=re.IGNORECASE)
        return text.strip()


def get_engine() -> LLMEngine:
    global _engine
    if _engine is None:
        _engine = LLMEngine()
    return _engine
