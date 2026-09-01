"""Tests for streaming support."""


from mlpal_assistants.transport.streaming import (
    ChatStreamDelta,
    StreamEvent,
)


class TestStreamEvent:
    """Tests for StreamEvent."""

    def test_basic_event(self):
        """Test creating basic event."""
        event = StreamEvent(
            event="message",
            data='{"content": "Hello"}',
        )

        assert event.event == "message"
        assert event.data == '{"content": "Hello"}'
        assert event.id is None
        assert event.retry is None

    def test_event_json_parsing(self):
        """Test parsing event data as JSON."""
        event = StreamEvent(
            event="message",
            data='{"content": "Hello", "role": "assistant"}',
        )

        data = event.json()

        assert data["content"] == "Hello"
        assert data["role"] == "assistant"

    def test_event_with_id_and_retry(self):
        """Test event with id and retry."""
        event = StreamEvent(
            event="message",
            data='{}',
            id="123",
            retry=5000,
        )

        assert event.id == "123"
        assert event.retry == 5000


class TestChatStreamDelta:
    """Tests for ChatStreamDelta."""

    def test_content_delta(self):
        """Test delta with content."""
        delta = ChatStreamDelta(content="Hello")

        assert delta.content == "Hello"
        assert delta.role is None
        assert delta.finish_reason is None

    def test_role_delta(self):
        """Test delta with role."""
        delta = ChatStreamDelta(role="assistant")

        assert delta.role == "assistant"
        assert delta.content is None

    def test_finish_delta(self):
        """Test delta with finish reason."""
        delta = ChatStreamDelta(finish_reason="stop")

        assert delta.finish_reason == "stop"
        assert delta.content is None

    def test_from_api_with_content(self):
        """Test parsing API response with content."""
        data = {
            "choices": [{
                "delta": {"content": "Hello"},
                "finish_reason": None,
            }]
        }

        delta = ChatStreamDelta.from_api(data)

        assert delta.content == "Hello"
        assert delta.finish_reason is None
        assert delta.raw == data

    def test_from_api_with_role(self):
        """Test parsing API response with role."""
        data = {
            "choices": [{
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }]
        }

        delta = ChatStreamDelta.from_api(data)

        assert delta.role == "assistant"

    def test_from_api_with_finish_reason(self):
        """Test parsing API response with finish reason."""
        data = {
            "choices": [{
                "delta": {},
                "finish_reason": "stop",
            }]
        }

        delta = ChatStreamDelta.from_api(data)

        assert delta.finish_reason == "stop"

    def test_from_api_with_tool_calls(self):
        """Test parsing API response with tool calls."""
        data = {
            "choices": [{
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": "call_1", "function": {"name": "get_weather"}}
                    ]
                },
                "finish_reason": None,
            }]
        }

        delta = ChatStreamDelta.from_api(data)

        assert delta.tool_calls is not None
        assert len(delta.tool_calls) == 1
        assert delta.tool_calls[0]["id"] == "call_1"

    def test_from_api_empty_choices(self):
        """Test parsing API response with empty choices."""
        data = {"choices": []}

        delta = ChatStreamDelta.from_api(data)

        assert delta.content is None
        assert delta.raw == data
