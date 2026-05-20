import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ImageInsertionRecord:
    image_id: str
    already_existed: bool


@dataclass
class IndexBatchParams:
    """
    Information necessary for efficiently indexing a batch of documents
    """

    image_id_to_previous_chunk_cnt: dict[str, int | None]
    image_id_to_new_chunk_cnt: dict[str, int]
    tenant_id: str
    large_chunks_enabled: bool

