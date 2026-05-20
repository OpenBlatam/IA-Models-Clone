"""
Callback Plugin Interface
=========================

Plugin interface for custom callbacks.

Author: BUL System
Date: 2024
"""

from typing import Any, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl
from .base_plugin import BasePlugin


class CallbackPlugin(BasePlugin, TrainerCallback):
    """
    Base class for callback plugins.
    
    Combines plugin functionality with TrainerCallback interface.
    
    Example:
        >>> class MyCallbackPlugin(CallbackPlugin):
        ...     def __init__(self):
        ...         super().__init__("my_callback", "1.0.0")
        ...     
        ...     def on_log(self, args, state, control, logs=None, **kwargs):
        ...         if self.enabled:
        ...             print(f"Step {state.global_step}: {logs}")
    """
    
    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs
    ) -> None:
        """Called when training logs are emitted."""
        if self.enabled:
            self.handle_log(args, state, control, logs, **kwargs)
    
    def handle_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs
    ) -> None:
        """
        Handle log event. Override this method.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            logs: Logs dictionary
            **kwargs: Additional arguments
        """
        pass

