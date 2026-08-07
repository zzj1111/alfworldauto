"""One place for machine-specific settings.

Precedence, highest first: caller environment > site file (.autoscaffold.env at the
repo root) > portable defaults. The shell resolver (autoscaffold/env.sh) reads the
same file with the same rules; test_config.py asserts the two parsers agree, because
a run whose shell half and python half disagree about a setting is incoherent.

Defaults must be derivable from the repo location, a well-known system path, or a
public identifier. No default may name a directory that exists on only one machine.
"""
from __future__ import annotations

import os
import re
import time

ENV_FILE_VAR = "ARM_ENV_FILE"
SITE_FILE_DEFAULT = ".autoscaffold.env"

_COMMENT = re.compile(r"\s#")


def repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("ARM_ROOT") or os.path.dirname(here)


def parse_value(raw):
    """The value half of a KEY=VALUE line, trailing comment removed.

    A quoted value keeps everything inside the quotes; an unquoted one ends at the
    first ` #` (a '#' with no space before it survives). Must match env.sh exactly.
    """
    v = raw.strip()
    if len(v) >= 2 and v[0] in "\"'":
        end = v.find(v[0], 1)
        return v[1:end] if end != -1 else v[1:]
    m = _COMMENT.search(v)
    return v[: m.start()].strip() if m else v


def load_site_file(path=None):
    """Read the site file into os.environ WITHOUT overriding what is already set."""
    path = path or os.environ.get(ENV_FILE_VAR) or os.path.join(repo_root(), SITE_FILE_DEFAULT)
    seen = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                    continue
                v = parse_value(v)
                seen[k] = v
                os.environ.setdefault(k, v)
    except OSError:
        return {}
    return seen


load_site_file()


def _get(name, default):
    return os.environ.get(name) or default


def exp_name():
    return _get("ARM_EXP", "alf_autoscaffold")


def workspace():
    return _get("ARM_WORKSPACE", os.path.join(repo_root(), "runs"))


def exp_root():
    return _get("ARM_EXP_ROOT", os.path.join(workspace(), "exp"))


def ckpt_root():
    return _get("ARM_CKPT_ROOT", os.path.join(workspace(), "ckpts"))


def log_dir():
    return _get("ARM_LOG_DIR", os.path.join(workspace(), "logs"))


def ray_tmp(exp=None):
    v = os.environ.get("ARM_RAY_TMP")
    if v:
        return v
    base = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) \
        else os.path.join(workspace(), "ray_tmp")
    return os.path.join(base, f"zray_{exp or exp_name()}")


def model_path():
    # A public HF identifier resolves on any machine with network or a cache;
    # a local path still overrides it.
    return _get("ARM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def alfworld_data():
    # alfworld's own default location (what `alfworld-download` writes). The vendored
    # env silently falls back here when ALFWORLD_DATA is unset, so the preflight must
    # verify the directory actually holds games — an empty one runs "0 games" quietly.
    return _get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))


def run_id():
    rid = os.environ.get("ARM_RUN_ID")
    if not rid:
        rid = time.strftime("%Y%m%d_%H%M%S")
        os.environ["ARM_RUN_ID"] = rid
    return rid


def stamped(path):
    """path with the launch id before the extension: a restart never appends into the
    previous attempt's file. Callers keep a `_latest` symlink for a stable tail target."""
    root, ext = os.path.splitext(path)
    return f"{root}_{run_id()}{ext}"


def container_memory():
    """(used_gb, limit_gb) for THIS container, or (used, total) on bare metal.

    /proc/meminfo inside a container reports the host, which passes exactly when it
    should fail; cgroup v2 then v1 first.
    """
    pairs = (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
             ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    for cur_f, max_f in pairs:
        try:
            with open(cur_f) as f:
                cur = int(f.read().strip())
            with open(max_f) as f:
                raw = f.read().strip()
            if raw == "max":
                break
            lim = int(raw)
            if lim >= 1 << 62:
                break
            return round(cur / 2 ** 30, 1), round(lim / 2 ** 30, 1)
        except (OSError, ValueError):
            continue
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.split()[0])
        total = info.get("MemTotal", 0) / 2 ** 20
        avail = info.get("MemAvailable", 0) / 2 ** 20
        return round(total - avail, 1), round(total, 1)
    except (OSError, ValueError):
        return None, None
