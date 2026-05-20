"""Factory stub for running celery worker / celery beat."""

from celery import Celery

from unified_core.utils.variable_functionality import fetch_versioned_implementation
from unified_core.utils.variable_functionality import set_is_ee_based_on_env_variable

set_is_ee_based_on_env_variable()
app: Celery = fetch_versioned_implementation(
    "unified_core.background.celery.apps.primary",
    "celery_app",
)
