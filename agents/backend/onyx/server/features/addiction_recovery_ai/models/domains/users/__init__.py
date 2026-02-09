"""
Users domain models
"""

from models.domains import register_model
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

try:
    from models.database import User, UserProfile
    
    def register_models():
        register_model("users", "User", User)
        register_model("users", "UserProfile", UserProfile)
except ImportError:
    pass



