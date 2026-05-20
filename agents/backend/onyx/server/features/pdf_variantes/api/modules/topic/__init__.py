"""
Topic Module
Complete module for topic extraction and management
"""

from .domain import TopicEntity, TopicFactory
from .application import (
    ExtractTopicsUseCase,
    GetTopicUseCase,
    ListTopicsUseCase
)
from .infrastructure import TopicRepository
from .presentation import TopicController, TopicPresenter

__all__ = [
    "TopicEntity",
    "TopicFactory",
    "ExtractTopicsUseCase",
    "GetTopicUseCase",
    "ListTopicsUseCase",
    "TopicRepository",
    "TopicController",
    "TopicPresenter"
]






