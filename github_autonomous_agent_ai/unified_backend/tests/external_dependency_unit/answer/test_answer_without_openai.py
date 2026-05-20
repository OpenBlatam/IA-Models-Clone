from __future__ import annotations

import os
from uuid import uuid4

from sqlalchemy.orm import Session

from unified_core.chat.models import AnswerStreamPart
from unified_core.chat.models import MessageResponseIDInfo
from unified_core.chat.models import StreamingError
from unified_core.chat.process_message import stream_chat_message_objects
from unified_core.context.search.models import RetrievalDetails
from unified_core.db.chat import create_chat_session
from unified_core.db.llm import fetch_existing_llm_providers
from unified_core.db.llm import remove_llm_provider
from unified_core.db.llm import update_default_provider
from unified_core.db.llm import upsert_llm_provider
from unified_core.server.manage.llm.models import LLMProviderUpsertRequest
from unified_core.server.manage.llm.models import ModelConfigurationUpsertRequest
from unified_core.server.query_and_chat.models import CreateChatMessageRequest
from unified_core.server.query_and_chat.streaming_models import MessageDelta
from unified_core.server.query_and_chat.streaming_models import MessageStart
from unified_core.server.query_and_chat.streaming_models import Packet
from tests.external_dependency_unit.conftest import create_test_user


def test_answer_with_only_anthropic_provider(
    db_session: Session,
    full_deployment_setup: None,
    mock_external_deps: None,
) -> None:
    """Ensure chat still streams answers when only an Anthropic provider is configured."""

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    assert anthropic_api_key, "ANTHROPIC_API_KEY environment variable must be set"

    # Drop any existing providers so that only Anthropic is available.
    for provider in fetch_existing_llm_providers(db_session):
        remove_llm_provider(db_session, provider.id)

    anthropic_model = "claude-3-5-sonnet-20240620"
    provider_name = f"anthropic-test-{uuid4().hex}"

    anthropic_provider = upsert_llm_provider(
        LLMProviderUpsertRequest(
            name=provider_name,
            provider="anthropic",
            api_key=anthropic_api_key,
            default_model_name=anthropic_model,
            fast_default_model_name=anthropic_model,
            is_public=True,
            groups=[],
            model_configurations=[
                ModelConfigurationUpsertRequest(name=anthropic_model, is_visible=True)
            ],
            api_key_changed=True,
        ),
        db_session=db_session,
    )

    try:
        update_default_provider(anthropic_provider.id, db_session)

        test_user = create_test_user(db_session, email_prefix="anthropic_only")
        chat_session = create_chat_session(
            db_session=db_session,
            description="Anthropic only chat",
            user_id=test_user.id,
            persona_id=0,
        )

        chat_request = CreateChatMessageRequest(
            chat_session_id=chat_session.id,
            parent_message_id=None,
            message="hello",
            file_descriptors=[],
            search_doc_ids=None,
            retrieval_options=RetrievalDetails(),
        )

        response_stream: list[AnswerStreamPart] = []
        for packet in stream_chat_message_objects(
            new_msg_req=chat_request,
            user=test_user,
            db_session=db_session,
        ):
            response_stream.append(packet)

        assert response_stream, "Should receive streamed packets"
        assert not any(
            isinstance(packet, StreamingError) for packet in response_stream
        ), "No streaming errors expected with Anthropic provider"

        has_message_id = any(
            isinstance(packet, MessageResponseIDInfo) for packet in response_stream
        )
        assert has_message_id, "Should include reserved assistant message ID"

        has_message_start = any(
            isinstance(packet, Packet) and isinstance(packet.obj, MessageStart)
            for packet in response_stream
        )
        assert has_message_start, "Stream should have a MessageStart packet"

        has_message_delta = any(
            isinstance(packet, Packet) and isinstance(packet.obj, MessageDelta)
            for packet in response_stream
        )
        assert has_message_delta, "Stream should have a MessageDelta packet"

    finally:
        remove_llm_provider(db_session, anthropic_provider.id)
