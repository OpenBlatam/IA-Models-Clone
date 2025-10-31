from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import msgspec
from typing import List, Optional

from typing import Any, List, Dict, Optional
import logging
import asyncio
class AnalyticsInfo(msgspec.Struct, frozen=True, slots=True):
    """
    Información de analytics y engagement.
    """
    analytics: Optional[dict] = None
    engagement_metrics: Optional[dict] = None
    auto_tags: List[str] = msgspec.field(default_factory=list)

    def with_analytics(self, analytics: dict) -> 'AnalyticsInfo':
        return self.update(analytics=analytics)

    def with_engagement_metrics(self, metrics: dict) -> 'AnalyticsInfo':
        return self.update(engagement_metrics=metrics)

    def with_auto_tags(self, tags: List[str]) -> 'AnalyticsInfo':
        return self.update(auto_tags=tags) 