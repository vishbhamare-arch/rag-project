import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from token_counter import TokenCounter
from token_tracker import TokenTracker


class TestIntegration:

    @pytest.fixture
    def setup(self):
        counter = TokenCounter()
        tracker = TokenTracker()
        return counter, tracker

    def test_full_workflow_single_query(self, setup):
        counter, tracker = setup

        doc_text = "This is a sample document about machine learning and AI."
        query = "What is machine learning?"
        response = "Machine learning is a subset of artificial intelligence that enables systems to learn and improve."

        doc_tokens = counter.count_tokens(doc_text)
        tracker.add_embedding_tokens(doc_tokens, "sample.pdf")

        query_tokens = counter.count_tokens(query)
        tracker.add_prompt_tokens(query_tokens, query)

        response_tokens = counter.count_tokens(response)
        tracker.add_response_tokens(response_tokens, response)

        tracker.add_message_tokens(
            message_id="msg1",
            role="assistant",
            content=response,
            prompt_tokens=query_tokens,
            response_tokens=response_tokens
        )

        summary = tracker.get_summary()
        assert summary["total_tokens"] > 0
        assert summary["embedding_tokens"] > 0
        assert summary["prompt_tokens"] > 0
        assert summary["response_tokens"] > 0
        assert summary["message_count"] == 1

    def test_multi_turn_conversation(self, setup):
        counter, tracker = setup

        documents = [
            "Python is a programming language.",
            "Java is another popular language.",
            "JavaScript is used for web development."
        ]

        total_doc_tokens = 0
        for doc in documents:
            tokens = counter.count_tokens(doc)
            tracker.add_embedding_tokens(tokens, f"doc_{documents.index(doc)}.pdf")
            total_doc_tokens += tokens

        queries = [
            "What is Python?",
            "Tell me about Java",
            "How is JavaScript used?"
        ]

        responses = [
            "Python is a high-level programming language.",
            "Java is a strongly-typed language.",
            "JavaScript is the language of the web."
        ]

        for i, (query, response) in enumerate(zip(queries, responses)):
            query_tokens = counter.count_tokens(query)
            response_tokens = counter.count_tokens(response)

            tracker.add_prompt_tokens(query_tokens, query)
            tracker.add_response_tokens(response_tokens, response)
            tracker.add_message_tokens(
                message_id=f"msg_{i}",
                role="assistant" if i % 2 == 1 else "user",
                content=response,
                prompt_tokens=query_tokens,
                response_tokens=response_tokens
            )

        summary = tracker.get_summary()
        assert summary["message_count"] == 3
        assert summary["embedding_tokens"] == total_doc_tokens
        assert summary["total_tokens"] > 0

    def test_tracker_persistence(self, setup):
        counter, tracker = setup

        tracker.add_embedding_tokens(100, "doc.pdf")
        tracker.add_prompt_tokens(50, "Query")
        tracker.add_response_tokens(200, "Response")
        tracker.add_message_tokens("msg1", "assistant", "Response", 50, 200)

        original_summary = tracker.get_summary()

        data = tracker.to_dict()
        restored_tracker = TokenTracker.from_dict(data)

        restored_summary = restored_tracker.get_summary()

        assert original_summary["total_tokens"] == restored_summary["total_tokens"]
        assert original_summary["embedding_tokens"] == restored_summary["embedding_tokens"]
        assert original_summary["prompt_tokens"] == restored_summary["prompt_tokens"]
        assert original_summary["response_tokens"] == restored_summary["response_tokens"]
        assert len(restored_summary["operations"]) == len(original_summary["operations"])

    def test_counter_with_various_text_formats(self, setup):
        counter, tracker = setup

        test_texts = [
            "Simple text",
            "Text with numbers: 12345",
            "Text with special chars: !@#$%",
            "Multi-line\ntext\nhere",
            "Text\twith\ttabs",
            "MixedCaseText",
            "UPPERCASE_TEXT",
            "lowercase_text"
        ]

        for text in test_texts:
            tokens = counter.count_tokens(text)
            assert tokens >= 0
            tracker.add_prompt_tokens(tokens, text)

        summary = tracker.get_summary()
        assert summary["prompt_tokens"] > 0

    def test_batch_document_processing(self, setup):
        counter, tracker = setup

        documents = [
            f"Document {i}: " + ("word " * 50)
            for i in range(10)
        ]

        tokens_list = counter.count_tokens_batch(documents)

        for i, tokens in enumerate(tokens_list):
            tracker.add_embedding_tokens(tokens, f"doc_{i}.pdf")

        summary = tracker.get_summary()
        assert summary["embedding_tokens"] > 0
        assert len(tokens_list) == 10

    def test_reset_and_reuse(self, setup):
        counter, tracker = setup

        tracker.add_embedding_tokens(100)
        tracker.add_prompt_tokens(50)
        assert tracker.total_tokens == 150

        tracker.reset()
        assert tracker.total_tokens == 0

        tracker.add_embedding_tokens(200)
        assert tracker.total_tokens == 200

    def test_operations_breakdown(self, setup):
        counter, tracker = setup

        for i in range(5):
            tracker.add_embedding_tokens(100, f"doc_{i}.pdf")

        for i in range(3):
            query = f"Query {i}"
            response = f"Response {i}"
            tracker.add_prompt_tokens(counter.count_tokens(query), query)
            tracker.add_response_tokens(counter.count_tokens(response), response)

        summary = tracker.get_summary()
        assert "embedding" in summary["operations"]
        assert "prompt" in summary["operations"]
        assert "response" in summary["operations"]


class TestTokenCounterIntegration:

    @pytest.fixture
    def counter(self):
        return TokenCounter()

    def test_estimate_and_count_consistency(self, counter):
        documents = ["doc 1 " * 10, "doc 2 " * 10, "doc 3 " * 10]

        individual_tokens = counter.count_tokens_batch(documents)
        estimated_total = counter.estimate_embedding_tokens(
            len(documents),
            counter.get_avg_tokens_per_doc(documents)
        )

        actual_total = sum(individual_tokens)
        assert estimated_total == actual_total

    def test_large_batch_processing(self, counter):
        documents = ["Sample document text. " * 10 for _ in range(100)]

        tokens_list = counter.count_tokens_batch(documents)
        list_total = counter.count_list_tokens(documents)

        assert len(tokens_list) == 100
        assert sum(tokens_list) == list_total

    def test_avg_tokens_calculation_accuracy(self, counter):
        documents = ["word " * 10, "word " * 20, "word " * 30]

        avg = counter.get_avg_tokens_per_doc(documents)
        individual_tokens = counter.count_tokens_batch(documents)
        expected_avg = sum(individual_tokens) / len(documents)

        assert avg == expected_avg


class TestComplexScenarios:

    @pytest.fixture
    def setup(self):
        counter = TokenCounter()
        tracker = TokenTracker()
        return counter, tracker

    def test_realistic_rag_workflow(self, setup):
        counter, tracker = setup

        pdf_content = "This document covers RAG (Retrieval-Augmented Generation). " * 50

        pdf_tokens = counter.count_tokens(pdf_content)
        tracker.add_embedding_tokens(pdf_tokens, "rag_paper.pdf")

        user_queries = [
            "What is RAG?",
            "How does retrieval augmentation work?",
            "What are the benefits of RAG?"
        ]

        for query in user_queries:
            query_tokens = counter.count_tokens(query)
            tracker.add_prompt_tokens(query_tokens, query)

            response = (
                "RAG is a technique that combines retrieval and generation. "
                "It retrieves relevant documents and uses them to augment the generation process. "
                "This improves accuracy and reduces hallucinations."
            )
            response_tokens = counter.count_tokens(response)
            tracker.add_response_tokens(response_tokens, response)

            tracker.add_message_tokens(
                message_id=f"query_{user_queries.index(query)}",
                role="assistant",
                content=response,
                prompt_tokens=query_tokens,
                response_tokens=response_tokens
            )

        summary = tracker.get_summary()
        breakdown = tracker.get_per_message_summary()

        assert summary["total_tokens"] > 0
        assert len(breakdown) == 3
        assert all(item["tokens"] > 0 for item in breakdown)

    def test_session_lifecycle(self, setup):
        counter, tracker = setup

        initial_summary = tracker.get_summary()
        assert initial_summary["total_tokens"] == 0

        tracker.add_embedding_tokens(1000, "doc.pdf")
        for i in range(5):
            tracker.add_prompt_tokens(50, f"Query {i}")
            tracker.add_response_tokens(100, f"Response {i}")

        active_summary = tracker.get_summary()
        assert active_summary["total_tokens"] > 1000

        tracker.reset()
        final_summary = tracker.get_summary()
        assert final_summary["total_tokens"] == 0

    def test_data_export_import(self, setup):
        counter, tracker = setup

        tracker.add_embedding_tokens(500, "document.pdf")
        for i in range(3):
            tracker.add_prompt_tokens(30, f"Q{i}")
            tracker.add_response_tokens(75, f"A{i}")
            tracker.add_message_tokens(f"msg_{i}", "assistant", f"Answer {i}", 30, 75)

        original_summary = tracker.get_summary()
        original_breakdown = tracker.get_message_breakdown()

        exported_data = tracker.to_dict()
        new_tracker = TokenTracker.from_dict(exported_data)

        new_summary = new_tracker.get_summary()
        new_breakdown = new_tracker.get_message_breakdown()

        assert original_summary["total_tokens"] == new_summary["total_tokens"]
        assert len(original_breakdown) == len(new_breakdown)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
