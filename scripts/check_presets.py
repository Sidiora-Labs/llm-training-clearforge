"""Validate shipped configuration files without requiring an accelerator."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clearforge.config import load_config  # noqa: E402

for path in sorted((ROOT / "configs").glob("*.toml")):
    config = load_config(path)
    assert config.data_dir.is_absolute()
    assert config.output_dir.is_absolute()
    print(f"validated {path.relative_to(ROOT)}")
