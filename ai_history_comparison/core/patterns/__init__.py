"""
Design Patterns Module

This module provides implementation of common design patterns
for the AI History Comparison System.
"""

from .singleton import Singleton
from .observer import Observer, Subject
from .strategy import Strategy, StrategyContext
from .factory import Factory, AbstractFactory
from .builder import Builder, Director
from .adapter import Adapter
from .decorator import Decorator
from .facade import Facade
from .proxy import Proxy
from .command import Command, CommandInvoker
from .chain_of_responsibility import Handler, ChainHandler
from .state import State, StateContext
from .template import TemplateMethod, AbstractTemplate

__all__ = [
    'Singleton',
    'Observer', 'Subject',
    'Strategy', 'StrategyContext',
    'Factory', 'AbstractFactory',
    'Builder', 'Director',
    'Adapter',
    'Decorator',
    'Facade',
    'Proxy',
    'Command', 'CommandInvoker',
    'Handler', 'ChainHandler',
    'State', 'StateContext',
    'TemplateMethod', 'AbstractTemplate'
]





















