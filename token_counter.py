from typing import List, Union
from transformers import AutoTokenizer

class TokenCounter:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            self.tokenizer = None
        self.model_name = model_name

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is None:
            return len(text.split())
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception:
            return len(text.split())

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        return [self.count_tokens(text) for text in texts]

    def estimate_embedding_tokens(self, num_docs: int, avg_tokens_per_doc: int) -> int:
        return num_docs * avg_tokens_per_doc

    def count_list_tokens(self, text_list: List[str]) -> int:
        return sum(self.count_tokens(text) for text in text_list)

    def get_avg_tokens_per_doc(self, texts: List[str]) -> float:
        if not texts:
            return 0
        total_tokens = self.count_list_tokens(texts)
        return total_tokens / len(texts)
