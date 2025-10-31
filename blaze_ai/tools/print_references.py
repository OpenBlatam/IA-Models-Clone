from __future__ import annotations

import json

from ..utils.references import get_references


def main() -> int:
    print(json.dumps(get_references(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


