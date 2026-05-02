from collections.abc import Generator

from ee.onyx.external_permissions.perm_sync_types import FetchAllDocumentsFunction
from ee.onyx.external_permissions.perm_sync_types import FetchAllDocumentsIdsFunction
from ee.onyx.external_permissions.utils import generic_doc_sync
from unified_core.access.models import DocExternalAccess
from unified_core.configs.constants import DocumentSource
from unified_core.connectors.sharepoint.connector import SharepointConnector
from unified_core.db.models import ConnectorCredentialPair
from unified_core.indexing.indexing_heartbeat import IndexingHeartbeatInterface
from unified_core.utils.logger import setup_logger

logger = setup_logger()

SHAREPOINT_DOC_SYNC_TAG = "sharepoint_doc_sync"


def sharepoint_doc_sync(
    cc_pair: ConnectorCredentialPair,
    fetch_all_existing_docs_fn: FetchAllDocumentsFunction,
    fetch_all_existing_docs_ids_fn: FetchAllDocumentsIdsFunction,
    callback: IndexingHeartbeatInterface | None = None,
) -> Generator[DocExternalAccess, None, None]:
    sharepoint_connector = SharepointConnector(
        **cc_pair.connector.connector_specific_config,
    )
    sharepoint_connector.load_credentials(cc_pair.credential.credential_json)

    yield from generic_doc_sync(
        cc_pair=cc_pair,
        fetch_all_existing_docs_ids_fn=fetch_all_existing_docs_ids_fn,
        callback=callback,
        doc_source=DocumentSource.SHAREPOINT,
        slim_connector=sharepoint_connector,
        label=SHAREPOINT_DOC_SYNC_TAG,
    )
