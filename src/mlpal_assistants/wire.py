"""Public re-export of the v2 wire types.

The native inference response (`Message`) shares its conventional name with the
v1 chat `Message` type exported at the package root, so it lives here rather
than at the top level:

    from mlpal_assistants.wire import Message, Usage, ContentBlock
"""

from __future__ import annotations

from mlpal_assistants._sse import ServerSentEvent
from mlpal_assistants._wire import (
    Catalog,
    ContentBlock,
    DailyUsage,
    DailyUsageBucket,
    FeedbackResult,
    KeyUsageSummary,
    ManagedKey,
    ManagedKeyList,
    ManagedKeyWithSecret,
    Message,
    ModelInfo,
    ModelUsage,
    Usage,
    UsageSummary,
)

__all__ = [
    "Message",
    "Usage",
    "ContentBlock",
    "ServerSentEvent",
    "ManagedKey",
    "ManagedKeyWithSecret",
    "ManagedKeyList",
    "KeyUsageSummary",
    "ModelInfo",
    "Catalog",
    "FeedbackResult",
    "UsageSummary",
    "ModelUsage",
    "DailyUsage",
    "DailyUsageBucket",
]
