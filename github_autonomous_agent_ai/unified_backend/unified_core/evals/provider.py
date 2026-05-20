from unified_core.evals.models import EvalProvider
from unified_core.evals.providers.braintrust import BraintrustEvalProvider


def get_default_provider() -> EvalProvider:
    return BraintrustEvalProvider()
