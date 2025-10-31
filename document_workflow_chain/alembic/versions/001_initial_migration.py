"""Initial migration

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create workflow_chains table
    op.create_table('workflow_chains',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('settings', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workflow_chains_id'), 'workflow_chains', ['id'], unique=False)
    op.create_index(op.f('ix_workflow_chains_name'), 'workflow_chains', ['name'], unique=False)
    op.create_index(op.f('ix_workflow_chains_status'), 'workflow_chains', ['status'], unique=False)
    op.create_index(op.f('ix_workflow_chains_created_at'), 'workflow_chains', ['created_at'], unique=False)
    op.create_index('idx_workflow_name_status', 'workflow_chains', ['name', 'status'], unique=False)
    op.create_index('idx_workflow_created_at', 'workflow_chains', ['created_at'], unique=False)
    
    # Create workflow_nodes table
    op.create_table('workflow_nodes',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('parent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('word_count', sa.Integer(), nullable=True),
        sa.Column('character_count', sa.Integer(), nullable=True),
        sa.Column('sentence_count', sa.Integer(), nullable=True),
        sa.Column('paragraph_count', sa.Integer(), nullable=True),
        sa.Column('reading_time_minutes', sa.Integer(), nullable=True),
        sa.Column('overall_score', sa.Integer(), nullable=True),
        sa.Column('readability_score', sa.Integer(), nullable=True),
        sa.Column('sentiment_score', sa.Integer(), nullable=True),
        sa.Column('seo_score', sa.Integer(), nullable=True),
        sa.Column('grammar_score', sa.Integer(), nullable=True),
        sa.Column('coherence_score', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['parent_id'], ['workflow_nodes.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['workflow_id'], ['workflow_chains.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workflow_nodes_id'), 'workflow_nodes', ['id'], unique=False)
    op.create_index(op.f('ix_workflow_nodes_workflow_id'), 'workflow_nodes', ['workflow_id'], unique=False)
    op.create_index(op.f('ix_workflow_nodes_title'), 'workflow_nodes', ['title'], unique=False)
    op.create_index(op.f('ix_workflow_nodes_parent_id'), 'workflow_nodes', ['parent_id'], unique=False)
    op.create_index(op.f('ix_workflow_nodes_priority'), 'workflow_nodes', ['priority'], unique=False)
    op.create_index(op.f('ix_workflow_nodes_status'), 'workflow_nodes', ['status'], unique=False)
    op.create_index(op.f('ix_workflow_nodes_created_at'), 'workflow_nodes', ['created_at'], unique=False)
    op.create_index('idx_node_workflow_priority', 'workflow_nodes', ['workflow_id', 'priority'], unique=False)
    op.create_index('idx_node_workflow_status', 'workflow_nodes', ['workflow_id', 'status'], unique=False)
    op.create_index('idx_node_parent', 'workflow_nodes', ['parent_id'], unique=False)
    op.create_index('idx_node_created_at', 'workflow_nodes', ['created_at'], unique=False)
    op.create_index('idx_node_title', 'workflow_nodes', ['title'], unique=False)
    
    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('roles', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('permissions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('avatar_url', sa.String(length=500), nullable=True),
        sa.Column('bio', sa.Text(), nullable=True),
        sa.Column('timezone', sa.String(length=50), nullable=True),
        sa.Column('language', sa.String(length=10), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)
    op.create_index(op.f('ix_users_is_active'), 'users', ['is_active'], unique=False)
    op.create_index(op.f('ix_users_created_at'), 'users', ['created_at'], unique=False)
    op.create_index('idx_user_active', 'users', ['is_active'], unique=False)
    op.create_index('idx_user_created_at', 'users', ['created_at'], unique=False)
    
    # Create audit_logs table
    op.create_table('audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('event_id', sa.String(length=100), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('session_id', sa.String(length=100), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('resource_type', sa.String(length=100), nullable=False),
        sa.Column('resource_id', sa.String(length=100), nullable=True),
        sa.Column('action', sa.String(length=50), nullable=False),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('result', sa.String(length=20), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_logs_id'), 'audit_logs', ['id'], unique=False)
    op.create_index(op.f('ix_audit_logs_event_type'), 'audit_logs', ['event_type'], unique=False)
    op.create_index(op.f('ix_audit_logs_event_id'), 'audit_logs', ['event_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_session_id'), 'audit_logs', ['session_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_ip_address'), 'audit_logs', ['ip_address'], unique=False)
    op.create_index(op.f('ix_audit_logs_resource_type'), 'audit_logs', ['resource_type'], unique=False)
    op.create_index(op.f('ix_audit_logs_resource_id'), 'audit_logs', ['resource_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_action'), 'audit_logs', ['action'], unique=False)
    op.create_index(op.f('ix_audit_logs_result'), 'audit_logs', ['result'], unique=False)
    op.create_index(op.f('ix_audit_logs_timestamp'), 'audit_logs', ['timestamp'], unique=False)
    op.create_index('idx_audit_event_type_timestamp', 'audit_logs', ['event_type', 'timestamp'], unique=False)
    op.create_index('idx_audit_user_timestamp', 'audit_logs', ['user_id', 'timestamp'], unique=False)
    op.create_index('idx_audit_resource_timestamp', 'audit_logs', ['resource_type', 'resource_id', 'timestamp'], unique=False)
    op.create_index('idx_audit_timestamp', 'audit_logs', ['timestamp'], unique=False)
    
    # Create system_metrics table
    op.create_table('system_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('metric_value', sa.Integer(), nullable=False),
        sa.Column('labels', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_system_metrics_id'), 'system_metrics', ['id'], unique=False)
    op.create_index(op.f('ix_system_metrics_metric_name'), 'system_metrics', ['metric_name'], unique=False)
    op.create_index(op.f('ix_system_metrics_metric_type'), 'system_metrics', ['metric_type'], unique=False)
    op.create_index(op.f('ix_system_metrics_timestamp'), 'system_metrics', ['timestamp'], unique=False)
    op.create_index('idx_metrics_name_timestamp', 'system_metrics', ['metric_name', 'timestamp'], unique=False)
    op.create_index('idx_metrics_timestamp', 'system_metrics', ['timestamp'], unique=False)
    
    # Add constraints
    op.create_check_constraint('ck_workflow_status', 'workflow_chains', "status IN ('created', 'active', 'paused', 'completed', 'error', 'cancelled', 'deleted')")
    op.create_check_constraint('ck_workflow_version', 'workflow_chains', 'version > 0')
    op.create_check_constraint('ck_node_priority', 'workflow_nodes', 'priority BETWEEN 1 AND 5')
    op.create_check_constraint('ck_node_status', 'workflow_nodes', "status IN ('created', 'active', 'paused', 'completed', 'error', 'cancelled', 'deleted')")
    op.create_check_constraint('ck_node_version', 'workflow_nodes', 'version > 0')
    op.create_check_constraint('ck_word_count', 'workflow_nodes', 'word_count >= 0')
    op.create_check_constraint('ck_character_count', 'workflow_nodes', 'character_count >= 0')
    op.create_check_constraint('ck_sentence_count', 'workflow_nodes', 'sentence_count >= 0')
    op.create_check_constraint('ck_paragraph_count', 'workflow_nodes', 'paragraph_count >= 0')
    op.create_check_constraint('ck_reading_time', 'workflow_nodes', 'reading_time_minutes >= 0')
    op.create_check_constraint('ck_overall_score', 'workflow_nodes', 'overall_score BETWEEN 0 AND 100')
    op.create_check_constraint('ck_readability_score', 'workflow_nodes', 'readability_score BETWEEN 0 AND 100')
    op.create_check_constraint('ck_sentiment_score', 'workflow_nodes', 'sentiment_score BETWEEN -100 AND 100')
    op.create_check_constraint('ck_seo_score', 'workflow_nodes', 'seo_score BETWEEN 0 AND 100')
    op.create_check_constraint('ck_grammar_score', 'workflow_nodes', 'grammar_score BETWEEN 0 AND 100')
    op.create_check_constraint('ck_coherence_score', 'workflow_nodes', 'coherence_score BETWEEN 0 AND 100')
    op.create_check_constraint('ck_username_length', 'users', 'length(username) >= 3')
    op.create_check_constraint('ck_email_format', 'users', "email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'")
    op.create_check_constraint('ck_audit_result', 'audit_logs', "result IN ('success', 'failure', 'error')")
    op.create_check_constraint('ck_audit_duration', 'audit_logs', 'duration_ms >= 0')
    op.create_check_constraint('ck_metric_type', 'system_metrics', "metric_type IN ('counter', 'gauge', 'histogram', 'summary')")


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('system_metrics')
    op.drop_table('audit_logs')
    op.drop_table('users')
    op.drop_table('workflow_nodes')
    op.drop_table('workflow_chains')




