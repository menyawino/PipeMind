from __future__ import annotations

import os
from pipemind.utils.audit import write_invocation_log


def test_write_invocation_log_uses_unique_timestamp(tmp_path):
    path1 = write_invocation_log(str(tmp_path), {"x": 1})
    path2 = write_invocation_log(str(tmp_path), {"x": 2})

    assert path1 != path2
    assert os.path.exists(path1)
    assert os.path.exists(path2)
