from typing import Dict, List, Optional
from datetime import datetime
import json


class TokenTracker:
    def __init__(self):
        self.total_embedding_tokens = 0
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.messages_token_usage: List[Dict] = []
        self.operations: Dict[str, int] = {}
        self.start_time = datetime.now()

    def add_embedding_tokens(self, count: int, doc_name: str = "unnamed") -> None:
        self.total_embedding_tokens += count
        if "embedding" not in self.operations:
            self.operations["embedding"] = 0
        self.operations["embedding"] += count

    def add_prompt_tokens(self, count: int, query: str = "") -> None:
        self.total_prompt_tokens += count
        if "prompt" not in self.operations:
            self.operations["prompt"] = 0
        self.operations["prompt"] += count

    def add_response_tokens(self, count: int, response: str = "") -> None:
        self.total_response_tokens += count
        if "response" not in self.operations:
            self.operations["response"] = 0
        self.operations["response"] += count

    def add_message_tokens(
        self,
        message_id: str,
        role: str,
        content: str,
        prompt_tokens: int = 0,
        response_tokens: int = 0
    ) -> None:
        self.messages_token_usage.append({
            "message_id": message_id,
            "role": role,
            "content": content[:100],
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total": prompt_tokens + response_tokens,
            "timestamp": datetime.now().isoformat()
        })

    def get_summary(self) -> Dict:
        return {
            "total_tokens": self.total_tokens,
            "embedding_tokens": self.total_embedding_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "response_tokens": self.total_response_tokens,
            "message_count": len(self.messages_token_usage),
            "operations": self.operations,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }

    def get_message_breakdown(self) -> List[Dict]:
        return self.messages_token_usage

    def get_per_message_summary(self) -> List[Dict]:
        summary = []
        for msg in self.messages_token_usage:
            summary.append({
                "role": msg["role"],
                "tokens": msg["total"],
                "timestamp": msg["timestamp"]
            })
        return summary

    @property
    def total_tokens(self) -> int:
        return (
            self.total_embedding_tokens +
            self.total_prompt_tokens +
            self.total_response_tokens
        )

    def reset(self) -> None:
        self.total_embedding_tokens = 0
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.messages_token_usage = []
        self.operations = {}
        self.start_time = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "total_embedding_tokens": self.total_embedding_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_response_tokens": self.total_response_tokens,
            "messages_token_usage": self.messages_token_usage,
            "operations": self.operations,
            "start_time": self.start_time.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TokenTracker":
        tracker = cls()
        tracker.total_embedding_tokens = data.get("total_embedding_tokens", 0)
        tracker.total_prompt_tokens = data.get("total_prompt_tokens", 0)
        tracker.total_response_tokens = data.get("total_response_tokens", 0)
        tracker.messages_token_usage = data.get("messages_token_usage", [])
        tracker.operations = data.get("operations", {})
        if "start_time" in data:
            tracker.start_time = datetime.fromisoformat(data["start_time"])
        return tracker

    def __repr__(self) -> str:
        return (
            f"TokenTracker(total={self.total_tokens}, "
            f"embedding={self.total_embedding_tokens}, "
            f"prompt={self.total_prompt_tokens}, "
            f"response={self.total_response_tokens})"
        )
