"""Configurable source locations, without requiring datasets merely to import code."""
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent


def mkg_directory():
    explicit=os.environ.get('TCM_MKG_DIR')
    if explicit:return Path(explicit).expanduser().resolve()
    candidates=[HERE/'data'/'TCM-MKG',HERE/'datasets'/'TCM-MKG']
    candidates.extend(part/'TCM-MKG' for part in sorted(HERE.parents[1].glob('01_*')))
    return next((path for path in candidates if path.is_dir()),candidates[0])


def inpetm_directory():
    explicit=os.environ.get('INPETM_DATA_DIR')
    return Path(explicit).expanduser().resolve() if explicit else HERE/'independent_data'/'inpetm'/'full_data'
