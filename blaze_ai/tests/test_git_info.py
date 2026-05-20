from __future__ import annotations

from pathlib import Path

from blaze_ai.utils.experiment import ExperimentTracker


def test_git_info_json_written(tmp_path: Path, monkeypatch) -> None:
    # Monkeypatch git calls to avoid requiring a repo in test env
    def fake_git_cmd(args):  # type: ignore[no-redef]
        if args[:2] == ["rev-parse", "HEAD"]:
            return "deadbeef"
        if args[:3] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return "main"
        if args[:2] == ["status", "--porcelain"]:
            return ""
        if args[:4] == ["config", "--get", "remote.origin.url"]:
            return "git@github.com:org/repo.git"
        return None

    monkeypatch.setattr(ExperimentTracker, "_git_cmd", staticmethod(fake_git_cmd))

    tracker = ExperimentTracker(tmp_path / "run_git")
    tracker.log_git_info()
    assert (tmp_path / "run_git" / "git.json").exists()


