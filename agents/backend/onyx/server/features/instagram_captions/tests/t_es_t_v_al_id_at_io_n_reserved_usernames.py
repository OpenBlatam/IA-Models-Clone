from ..utils.validation import validate_instagram_username


def test_reserved_usernames_rejected():
    for name in ["admin", "instagram", "meta", "facebook", "help", "support"]:
        res = validate_instagram_username(username=name)
        assert res["is_valid"] is False
        assert res["error"] == "Username is reserved"



