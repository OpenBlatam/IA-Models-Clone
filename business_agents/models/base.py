"""
Base Model
==========

Base model and common database components.
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, JSON, ForeignKey, Table, Index
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

# Association tables for many-to-many relationships
workflow_agents = Table(
    'workflow_agents',
    Base.metadata,
    Column('workflow_id', String, ForeignKey('workflows.id'), primary_key=True),
    Column('agent_id', String, ForeignKey('business_agents.id'), primary_key=True)
)

workflow_templates = Table(
    'workflow_templates',
    Base.metadata,
    Column('workflow_id', String, ForeignKey('workflows.id'), primary_key=True),
    Column('template_id', String, ForeignKey('templates.id'), primary_key=True)
)

user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', String, ForeignKey('users.id'), primary_key=True),
    Column('role_id', String, ForeignKey('roles.id'), primary_key=True)
)
