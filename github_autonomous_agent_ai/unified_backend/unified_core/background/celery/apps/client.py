from celery import Celery

import unified_core.background.celery.apps.app_base as app_base

celery_app = Celery(__name__)
celery_app.config_from_object("unified_core.background.celery.configs.client")
celery_app.Task = app_base.TenantAwareTask  # type: ignore [misc]
