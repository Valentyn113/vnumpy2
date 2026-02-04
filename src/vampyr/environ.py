from os import environ
from pathlib import Path
from sysconfig import get_path


def _set_mwfilters_path():
    """
    Sets location of filter files.
    """

    if "MWFILTERS_DIR" not in environ:
        # Check standard system/venv prefix location
        # e.g. /usr/share or /path/to/venv/share
        site_packages = Path(get_path("purelib"))
        venv_root = site_packages.parents[2]

        candidates = [
            venv_root / "share/MRCPP/mwfilters",          # Conda / System
            site_packages / "share/MRCPP/mwfilters",      # Wheels / pip
            Path(__file__).parent.parent / "share/MRCPP/mwfilters" # Relative fallback
        ]

        for p in candidates:
            if p.exists():
                environ["MWFILTERS_DIR"] = str(p.resolve())
                return

        # Default fallback to venv root (original behavior)
        environ["MWFILTERS_DIR"] = str(venv_root / "share/MRCPP/mwfilters")
