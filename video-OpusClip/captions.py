from agents.backend.onyx.server.features.utils import OnyxBaseModel, validate_model, cache_model, log_operations, ModelStatus
from typing import Dict, List, Optional
from datetime import datetime
from onyx.utils.logger import setup_logger
from onyx.utils.langchain import LangchainField
import structlog
from prometheus_client import Counter

logger = setup_logger()
log = structlog.get_logger()

class VideoInput(OnyxBaseModel):
    """Input model for video processing following Onyx conventions"""
    youtube_url: str
    target_duration: Optional[int] = 60
    style_preferences: Optional[Dict[str, str]] = {}
    brand_kit: Optional[Dict[str, any]] = {}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: ModelStatus = ModelStatus.ACTIVE

    @classmethod
    def batch_to_dicts(cls, objs: List["VideoInput"]) -> List[dict]:
        Counter('videoinput_batch_to_dicts_total', 'Total batch_to_dicts calls').inc()
        log.info("batch_to_dicts", count=len(objs))
        return [obj.to_dict() for obj in objs]

    @classmethod
    def batch_from_dicts(cls, dicts: List[dict]) -> List["VideoInput"]:
        Counter('videoinput_batch_from_dicts_total', 'Total batch_from_dicts calls').inc()
        log.info("batch_from_dicts", count=len(dicts))
        return [cls.from_dict(d) for d in dicts]

    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="youtube_url")
    @log_operations()
    def save(self, user_context=None):
        super().save(user_context=user_context)

class VideoOutput(OnyxBaseModel):
    """Output model for processed video following Onyx conventions"""
    short_url: str
    duration: int
    segments: List[Dict[str, any]]
    captions: str
    metadata: Dict[str, any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: ModelStatus = ModelStatus.ACTIVE

    @classmethod
    def batch_to_dicts(cls, objs: List["VideoOutput"]) -> List[dict]:
        Counter('videooutput_batch_to_dicts_total', 'Total batch_to_dicts calls').inc()
        log.info("batch_to_dicts", count=len(objs))
        return [obj.to_dict() for obj in objs]

    @classmethod
    def batch_from_dicts(cls, dicts: List[dict]) -> List["VideoOutput"]:
        Counter('videooutput_batch_from_dicts_total', 'Total batch_from_dicts calls').inc()
        log.info("batch_from_dicts", count=len(dicts))
        return [cls.from_dict(d) for d in dicts]

    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="short_url")
    @log_operations()
    def save(self, user_context=None):
        super().save(user_context=user_context)

class VideoState(OnyxBaseModel):
    """State model for video processing workflow following Onyx conventions"""
    messages: List = []
    input: Optional[VideoInput] = None
    output: Optional[VideoOutput] = None
    error: Optional[str] = None
    current_step: str = "start"
    processing_status: Dict[str, any] = {}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: ModelStatus = ModelStatus.ACTIVE

    @classmethod
    def batch_to_dicts(cls, objs: List["VideoState"]) -> List[dict]:
        Counter('videostate_batch_to_dicts_total', 'Total batch_to_dicts calls').inc()
        log.info("batch_to_dicts", count=len(objs))
        return [obj.to_dict() for obj in objs]

    @classmethod
    def batch_from_dicts(cls, dicts: List[dict]) -> List["VideoState"]:
        Counter('videostate_batch_from_dicts_total', 'Total batch_from_dicts calls').inc()
        log.info("batch_from_dicts", count=len(dicts))
        return [cls.from_dict(d) for d in dicts]

    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="current_step")
    @log_operations()
    def save(self, user_context=None):
        super().save(user_context=user_context)

# Batch methods for VideoInput/Output/State can be added as needed, following the same pattern as in models.py 