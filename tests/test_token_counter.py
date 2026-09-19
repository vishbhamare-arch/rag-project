import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from token_counter import TokenCounter


class TestTokenCounter:

    @pytest.fixture
    def counter(self):
        return TokenCounter()

    def test_counter_initialization(self, counter):
        assert counter is not None
        assert counter.model_name == "meta-llama/Llama-2-7b-hf"

    def test_count_empty_text(self, counter):
        result = counter.count_tokens("")
        assert result == 0

    def test_count_single_word(self, counter):
        result = counter.count_tokens("hello")
        assert result > 0

    def test_count_multiple_words(self, counter):
        text = "This is a test sentence with multiple words"
        result = counter.count_tokens(text)
        assert result > 5

    def test_count_longer_text(self, counter):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        result = counter.count_tokens(text)
        assert result > 50

    def test_count_tokens_consistency(self, counter):
        text = "Hello world"
        result1 = counter.count_tokens(text)
        result2 = counter.count_tokens(text)
        assert result1 == result2

    def test_count_tokens_batch(self, counter):
        texts = ["hello", "world", "test"]
        results = counter.count_tokens_batch(texts)
        assert len(results) == 3
        assert all(r > 0 for r in results)

    def test_count_tokens_batch_with_empty(self, counter):
        texts = ["hello", "", "test"]
        results = counter.count_tokens_batch(texts)
        assert len(results) == 3
        assert results[1] == 0

    def test_estimate_embedding_tokens(self, counter):
        num_docs = 5
        avg_tokens = 100
        result = counter.estimate_embedding_tokens(num_docs, avg_tokens)
        assert result == 500

    def test_estimate_embedding_tokens_zero(self, counter):
        result = counter.estimate_embedding_tokens(0, 100)
        assert result == 0

    def test_count_list_tokens(self, counter):
        texts = ["hello", "world", "test"]
        result = counter.count_list_tokens(texts)
        assert result > 0

    def test_count_list_tokens_empty(self, counter):
        texts = []
        result = counter.count_list_tokens(texts)
        assert result == 0

    def test_get_avg_tokens_per_doc(self, counter):
        texts = ["hello world", "test sentence", "another example"]
        result = counter.get_avg_tokens_per_doc(texts)
        assert result > 0
        assert isinstance(result, float)

    def test_get_avg_tokens_per_doc_empty(self, counter):
        texts = []
        result = counter.get_avg_tokens_per_doc(texts)
        assert result == 0

    def test_get_avg_tokens_per_doc_single(self, counter):
        texts = ["hello world"]
        result = counter.get_avg_tokens_per_doc(texts)
        assert result > 0

    def test_special_characters(self, counter):
        text = "Hello! @#$% &*() <tag>test</tag>"
        result = counter.count_tokens(text)
        assert result > 0

    def test_unicode_characters(self, counter):
        text = "Hello 世界 مرحبا мир"
        result = counter.count_tokens(text)
        assert result > 0

    def test_very_long_text(self, counter):
        text = "word " * 10000
        result = counter.count_tokens(text)
        assert result > 1000

    def test_newlines_and_spaces(self, counter):
        text = "line1\nline2\nline3\n\nline4"
        result = counter.count_tokens(text)
        assert result > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
