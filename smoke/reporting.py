from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORTS_DIR = Path("/workspace/reports")


def write_report(report_name: str, payload: dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{report_name}.json"
    enriched_payload = {
        "report_name": report_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        **payload,
    }
    report_path.write_text(
        json.dumps(enriched_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"report_path={report_path}")
    return report_path
