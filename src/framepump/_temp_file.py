"""Atomic file writing via temp file + rename."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


class TempFile:
    """Write to a temporary file, then rename to final path on success.

    Usage:
        tf = TempFile('output.mp4')
        with open(tf.temp_path, 'wb') as f:
            f.write(data)
        tf.finalize()  # rename temp → final

        # Or on error:
        tf.cleanup()  # delete temp
    """

    def __init__(self, path: PathLike) -> None:
        self.final_path = Path(path)
        suffix = f'.tmp_{secrets.token_hex(4)}'
        self.temp_path = self.final_path.with_suffix(suffix)

    def finalize(self) -> None:
        """Rename temp file to final path."""
        os.replace(self.temp_path, self.final_path)

    def cleanup(self) -> None:
        """Delete temp file."""
        self.temp_path.unlink(missing_ok=True)