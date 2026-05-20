"""
Scheduling and task management utilities
"""

from utils.categories import register_utility

try:
    from utils.scheduler import Scheduler
    from utils.schedulers import Schedulers
    from utils.message_queue import MessageQueue
    from utils.queue_utils import QueueUtils
    
    def register_utilities():
        register_utility("scheduling", "scheduler", Scheduler)
        register_utility("scheduling", "schedulers", Schedulers)
        register_utility("scheduling", "message_queue", MessageQueue)
        register_utility("scheduling", "queue_utils", QueueUtils)
except ImportError:
    pass



