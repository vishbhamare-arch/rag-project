import pytest
import sys
import os
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from token_tracker import TokenTracker


class TestTokenTracker:

    @pytest.fixture
    def tracker(self):
        return TokenTracker()

    def test_tracker_initialization(self, tracker):
        assert tracker.total_embedding_tokens == 0
        assert tracker.total_prompt_tokens == 0
        assert tracker.total_response_tokens == 0
        assert tracker.total_tokens == 0
        assert len(tracker.messages_token_usage) == 0
        assert len(tracker.operations) == 0

    def test_add_embedding_tokens(self, tracker):
        tracker.add_embedding_tokens(100, "test.pdf")
        assert tracker.total_embedding_tokens == 100
        assert tracker.operations["embedding"] == 100

    def test_add_embedding_tokens_multiple(self, tracker):
        tracker.add_embedding_tokens(100, "test1.pdf")
        tracker.add_embedding_tokens(50, "test2.pdf")
        assert tracker.total_embedding_tokens == 150
        assert tracker.operations["embedding"] == 150

    def test_add_prompt_tokens(self, tracker):
        tracker.add_prompt_tokens(50, "What is this?")
        assert tracker.total_prompt_tokens == 50
        assert tracker.operations["prompt"] == 50

    def test_add_prompt_tokens_multiple(self, tracker):
        tracker.add_prompt_tokens(50, "Query 1")
        tracker.add_prompt_tokens(30, "Query 2")
        assert tracker.total_prompt_tokens == 80
        assert tracker.operations["prompt"] == 80

    def test_add_response_tokens(self, tracker):
        tracker.add_response_tokens(200, "This is the answer...")
        assert tracker.total_response_tokens == 200
        assert tracker.operations["response"] == 200

    def test_add_response_tokens_multiple(self, tracker):
        tracker.add_response_tokens(200, "Response 1")
        tracker.add_response_tokens(150, "Response 2")
        assert tracker.total_response_tokens == 350
        assert tracker.operations["response"] == 350

    def test_total_tokens_property(self, tracker):
        tracker.add_embedding_tokens(100)
        tracker.add_prompt_tokens(50)
        tracker.add_response_tokens(200)
        assert tracker.total_tokens == 350

    def test_add_message_tokens(self, tracker):
        tracker.add_message_tokens(
            message_id="msg1",
            role="user",
            content="Test query",
            prompt_tokens=10,
            response_tokens=50
        )
        assert len(tracker.messages_token_usage) == 1
        msg = tracker.messages_token_usage[0]
        assert msg["message_id"] == "msg1"
        assert msg["role"] == "user"
        assert msg["total"] == 60

    def test_add_message_tokens_multiple(self, tracker):
        tracker.add_message_tokens("msg1", "user", "Query 1", 10, 50)
        tracker.add_message_tokens("msg2", "assistant", "Answer 1", 0, 100)
        tracker.add_message_tokens("msg3", "user", "Query 2", 15, 0)
        assert len(tracker.messages_token_usage) == 3

    def test_get_summary(self, tracker):
        tracker.add_embedding_tokens(100)
        tracker.add_prompt_tokens(50)
        tracker.add_response_tokens(200)
        tracker.add_message_tokens("msg1", "user", "test", 50, 200)

        summary = tracker.get_summary()
        assert summary["total_tokens"] == 350
        assert summary["embedding_tokens"] == 100
        assert summary["prompt_tokens"] == 50
        assert summary["response_tokens"] == 200
        assert summary["message_count"] == 1

    def test_get_message_breakdown(self, tracker):
        tracker.add_message_tokens("msg1", "user", "Query", 10, 50)
        tracker.add_message_tokens("msg2", "assistant", "Answer", 0, 100)

        breakdown = tracker.get_message_breakdown()
        assert len(breakdown) == 2
        assert breakdown[0]["role"] == "user"
        assert breakdown[1]["role"] == "assistant"

    def test_get_per_message_summary(self, tracker):
        tracker.add_message_tokens("msg1", "user", "Query", 10, 50)
        tracker.add_message_tokens("msg2", "assistant", "Answer", 0, 100)

        summary = tracker.get_per_message_summary()
        assert len(summary) == 2
        assert summary[0]["role"] == "user"
        assert summary[0]["tokens"] == 60
        assert summary[1]["role"] == "assistant"
        assert summary[1]["tokens"] == 100

    def test_reset(self, tracker):
        tracker.add_embedding_tokens(100)
        tracker.add_prompt_tokens(50)
        tracker.add_response_tokens(200)
        tracker.add_message_tokens("msg1", "user", "test", 50, 200)

        tracker.reset()
        assert tracker.total_embedding_tokens == 0
        assert tracker.total_prompt_tokens == 0
        assert tracker.total_response_tokens == 0
        assert len(tracker.messages_token_usage) == 0
        assert len(tracker.operations) == 0

    def test_to_dict(self, tracker):
        tracker.add_embedding_tokens(100)
        tracker.add_prompt_tokens(50)
        tracker.add_response_tokens(200)

        data = tracker.to_dict()
        assert data["total_embedding_tokens"] == 100
        assert data["total_prompt_tokens"] == 50
        assert data["total_response_tokens"] == 200
        assert "start_time" in data

    def test_from_dict(self):
        original = TokenTracker()
        original.add_embedding_tokens(100)
        original.add_prompt_tokens(50)
        original.add_response_tokens(200)

        data = original.to_dict()
        restored = TokenTracker.from_dict(data)

        assert restored.total_embedding_tokens == 100
        assert restored.total_prompt_tokens == 50
        assert restored.total_response_tokens == 200

    def test_to_dict_from_dict_roundtrip(self, tracker):
        tracker.add_embedding_tokens(100)
        tracker.add_prompt_tokens(50)
        tracker.add_response_tokens(200)
        tracker.add_message_tokens("msg1", "user", "test", 50, 200)

        data = tracker.to_dict()
        restored = TokenTracker.from_dict(data)

        assert restored.total_tokens == tracker.total_tokens
        assert len(restored.messages_token_usage) == len(tracker.messages_token_usage)

    def test_repr(self, tracker):
        tracker.add_embedding_tokens(100)
        tracker.add_prompt_tokens(50)
        tracker.add_response_tokens(200)

        repr_str = repr(tracker)
        assert "total=350" in repr_str
        assert "embedding=100" in repr_str

    def test_message_token_usage_content_truncation(self, tracker):
        long_content = "x" * 500
        tracker.add_message_tokens("msg1", "user", long_content, 10, 50)

        msg = tracker.messages_token_usage[0]
        assert len(msg["content"]) == 100

    def test_operations_aggregation(self, tracker):
        tracker.add_embedding_tokens(50)
        tracker.add_embedding_tokens(30)
        tracker.add_prompt_tokens(20)
        tracker.add_prompt_tokens(15)
        tracker.add_response_tokens(100)

        assert tracker.operations["embedding"] == 80
        assert tracker.operations["prompt"] == 35
        assert tracker.operations["response"] == 100

    def test_uptime_seconds_in_summary(self, tracker):
        import time
        time.sleep(0.1)
        summary = tracker.get_summary()
        assert "uptime_seconds" in summary
        assert summary["uptime_seconds"] >= 0.1

    def test_timestamp_in_messages(self, tracker):
        tracker.add_message_tokens("msg1", "user", "test", 10, 50)
        msg = tracker.messages_token_usage[0]
        assert "timestamp" in msg
        timestamp = datetime.fromisoformat(msg["timestamp"])
        assert isinstance(timestamp, datetime)


class TestTokenTrackerEdgeCases:

    @pytest.fixture
    def tracker(self):
        return TokenTracker()

    def test_zero_token_additions(self, tracker):
        tracker.add_embedding_tokens(0)
        tracker.add_prompt_tokens(0)
        tracker.add_response_tokens(0)
        assert tracker.total_tokens == 0

    def test_large_token_counts(self, tracker):
        tracker.add_embedding_tokens(1000000)
        tracker.add_prompt_tokens(500000)
        tracker.add_response_tokens(500000)
        assert tracker.total_tokens == 2000000

    def test_negative_token_counts_not_prevented(self, tracker):
        tracker.add_embedding_tokens(-100)
        assert tracker.total_embedding_tokens == -100

    def test_multiple_message_additions(self, tracker):
        for i in range(100):
            tracker.add_message_tokens(f"msg{i}", "user" if i % 2 == 0 else "assistant", f"content{i}", 10, 20)
        assert len(tracker.messages_token_usage) == 100

    def test_empty_operations_initially(self, tracker):
        summary = tracker.get_summary()
        assert summary["operations"] == {}

    def test_summary_with_no_messages(self, tracker):
        summary = tracker.get_summary()
        assert summary["message_count"] == 0
        assert summary["total_tokens"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
