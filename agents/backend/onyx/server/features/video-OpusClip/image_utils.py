from typing import Tuple
from sqlalchemy.orm import Session
from onyx.utils.logger import setup_logger
import structlog
from agents.backend.onyx.server.features.utils import OnyxBaseModel, log_operations

logger = setup_logger()
log = structlog.get_logger()

class ImageUtils(OnyxBaseModel):
    """Image utility methods for Onyx video pipeline."""
    @log_operations()
    def store_image_and_create_section(
        self,
        db_session: Session,
        image_data: bytes,
        file_name: str,
        display_name: str,
        link: str | None = None,
        media_type: str = "application/octet-stream",
        file_origin = None,
    ) -> Tuple:
        stored_file_name = None
        try:
            pgfilestore = save_bytes_to_pgfilestore(
                db_session=db_session,
                raw_bytes=image_data,
                media_type=media_type,
                identifier=file_name,
                display_name=display_name,
                file_origin=file_origin,
            )
            stored_file_name = pgfilestore.file_name
        except Exception as e:
            logger.error(f"Failed to store image: {e}")
            self._log_audit("store_image_error", {"error": str(e), "file_name": file_name})
            raise e
        return (
            ImageSection(image_file_name=stored_file_name, link=link),
            stored_file_name,
        ) 