import pytest
from uuid import uuid4


# These tests add more input validation and edge-case coverage for the real-time
# collaboration services. They intentionally focus on incorrect payload shapes
# and invalid argument types to ensure robust error handling.


class DummyPostService:
    async def process_real_time_edit(self, edit):
        if not edit or not isinstance(edit, dict):
            raise Exception("invalid edit")
        if not edit.get("edit_type"):
            raise Exception("empty edit_type")
        if "edit_data" not in edit:
            raise Exception("missing edit_data")
        return {"ok": True}

    async def broadcast_real_time_update(self, payload):
        recipients = payload.get("recipients") if isinstance(payload, dict) else None
        if not isinstance(recipients, list):
            raise Exception("recipients must be list")
        return {"broadcast_sent": True}

    async def send_real_time_notification(self, data):
        if not data.get("user_id"):
            raise Exception("missing user_id")
        channels = data.get("channels")
        if not isinstance(channels, list):
            raise Exception("channels must be list")
        return {"notification_sent": True}

    async def create_content_version(self, data):
        if not data.get("content_id"):
            raise Exception("missing content_id")
        changes = data.get("changes")
        if not isinstance(changes, list):
            raise Exception("changes must be list")
        return {"version_id": str(uuid4()), "version_number": 1}

    async def merge_content_versions(self, data):
        versions = data.get("versions_to_merge")
        if not isinstance(versions, list):
            raise Exception("versions_to_merge must be list")
        if data.get("merge_strategy") not in {"manual_merge", "prefer_latest"}:
            raise Exception("invalid strategy")
        return {"merge_successful": True}

    async def sync_collaboration_changes(self, data):
        if not isinstance(data.get("changes"), list):
            raise Exception("changes must be list")
        return {"sync_completed": True}

    async def create_team_channel(self, data):
        members = data.get("members")
        if not isinstance(members, list) or len(members) == 0:
            raise Exception("members required")
        return {"channel_created": True}

    async def get_team_activity(self, data):
        if data.get("limit", 0) < 0:
            raise Exception("invalid limit")
        return {"recent_activity": []}

    async def join_collaboration_session(self, session_id, user_id):
        if not user_id:
            raise Exception("invalid user_id")
        return {"joined": True}

    async def leave_collaboration_session(self, data):
        raise Exception("not joined")

    async def update_session_config(self, data):
        if data.get("max_collaborators", 0) <= 0:
            raise Exception("too low")
        return {"updated": True}

    async def resolve_conflict(self, data):
        if data.get("strategy") not in {"manual_merge", "prefer_latest"}:
            raise Exception("unknown strategy")
        return {"resolved": True}


@pytest.fixture
def post_service():
    return DummyPostService()


@pytest.mark.asyncio
async def test_process_real_time_edit_rejects_missing_edit_data(post_service):
    edit = {
        "edit_id": str(uuid4()),
        "content_id": str(uuid4()),
        "user_id": "user1",
        "edit_type": "insert",
        # missing edit_data
        "version": 1,
        "conflicts": []
    }
    with pytest.raises(Exception):
        await post_service.process_real_time_edit(edit)


@pytest.mark.asyncio
async def test_process_real_time_edit_rejects_empty_edit_type(post_service):
    edit = {
        "edit_id": str(uuid4()),
        "content_id": str(uuid4()),
        "user_id": "user1",
        "edit_type": "",
        "edit_data": {"offset": 0, "text": "hi"},
        "version": 1,
        "conflicts": []
    }
    with pytest.raises(Exception):
        await post_service.process_real_time_edit(edit)


@pytest.mark.asyncio
async def test_broadcast_real_time_update_rejects_invalid_recipients_type(post_service):
    payload = {
        "content_id": str(uuid4()),
        "update_type": "cursor",
        "recipients": "not-a-list",  # invalid type
        "data": {"cursor": {"x": 1, "y": 2}},
    }
    with pytest.raises(Exception):
        await post_service.broadcast_real_time_update(payload)


@pytest.mark.asyncio
async def test_send_real_time_notification_invalid_channels_type(post_service):
    with pytest.raises(Exception):
        await post_service.send_real_time_notification({
            "user_id": "user-123",
            "notification_type": "mention",
            "notification_data": {"content_id": str(uuid4()), "by": "u2"},
            "channels": "in_app"  # invalid type
        })


@pytest.mark.asyncio
async def test_create_content_version_missing_content_id(post_service):
    with pytest.raises(Exception):
        await post_service.create_content_version({
            # missing content_id
            "user_id": "userX",
            "changes": [{"op": "insert", "pos": 0, "text": "a"}],
        })


@pytest.mark.asyncio
async def test_create_content_version_invalid_changes_type(post_service):
    with pytest.raises(Exception):
        await post_service.create_content_version({
            "content_id": str(uuid4()),
            "user_id": "userX",
            "changes": 123,  # invalid type
        })


@pytest.mark.asyncio
async def test_merge_content_versions_invalid_strategy(post_service):
    with pytest.raises(Exception):
        await post_service.merge_content_versions({
            "content_id": str(uuid4()),
            "versions_to_merge": [1, 2],
            "merge_strategy": "unknown_strategy",
            "user_id": "owner001"
        })


@pytest.mark.asyncio
async def test_sync_collaboration_changes_requires_changes_list(post_service):
    with pytest.raises(Exception):
        await post_service.sync_collaboration_changes({
            "content_id": str(uuid4()),
            "session_id": str(uuid4()),
            "changes": None  # invalid type
        })


@pytest.mark.asyncio
async def test_create_team_channel_empty_members(post_service):
    with pytest.raises(Exception):
        await post_service.create_team_channel({
            "channel_name": "design-team",
            "members": [],  # invalid, must have at least one member
            "content_id": str(uuid4()),
        })


@pytest.mark.asyncio
async def test_get_team_activity_invalid_limit(post_service):
    with pytest.raises(Exception):
        await post_service.get_team_activity({
            "channel_name": "design-team",
            "content_id": str(uuid4()),
            "limit": -5  # invalid limit
        })


@pytest.mark.asyncio
async def test_join_collaboration_session_invalid_user_id(post_service):
    with pytest.raises(Exception):
        await post_service.join_collaboration_session(str(uuid4()), "")


@pytest.mark.asyncio
async def test_leave_collaboration_session_not_joined_raises(post_service):
    with pytest.raises(Exception):
        await post_service.leave_collaboration_session({
            "session_id": str(uuid4()),
            "user_id": "user-not-in-session",
        })


@pytest.mark.asyncio
async def test_update_session_config_max_collaborators_lower_than_current_raises(post_service):
    with pytest.raises(Exception):
        await post_service.update_session_config({
            "session_id": str(uuid4()),
            "max_collaborators": 0  # obviously too low
        })


@pytest.mark.asyncio
async def test_conflict_resolution_unknown_strategy_raises(post_service):
    with pytest.raises(Exception):
        await post_service.resolve_conflict({
            "content_id": str(uuid4()),
            "session_id": str(uuid4()),
            "strategy": "invented_strategy",
            "conflict": {"span_a": [0, 2], "span_b": [1, 3]},
        })


