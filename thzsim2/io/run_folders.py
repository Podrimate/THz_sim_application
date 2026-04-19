from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re


@dataclass(slots=True)
class RunFolders:
    run_id: str
    created_at: str
    run_dir: Path
    reference_dir: Path


def slugify(text: str | None):
    raw = "reference" if text is None else str(text).strip()
    if not raw:
        raw = "reference"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").lower()
    return slug or "reference"


def create_reference_run_folders(output_root="runs", run_label=None, now: datetime | None = None):
    output_root = Path(output_root)
    current = datetime.now().astimezone() if now is None else now.astimezone()
    slug = slugify(run_label)

    while True:
        stamp = current.strftime("%Y%m%d_%H%M%S")
        run_id = f"{stamp}__reference__{slug}"
        run_dir = output_root / run_id
        if not run_dir.exists():
            break
        current = current + timedelta(seconds=1)

    reference_dir = run_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=False)
    return RunFolders(
        run_id=run_id,
        created_at=current.isoformat(),
        run_dir=run_dir,
        reference_dir=reference_dir,
    )
