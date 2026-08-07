import os
import tempfile

from autoscaffold import config as C


def test_parse_value_strips_trailing_comments_but_keeps_quoted_hashes():
    cases = {
        "10        # training steps per cycle": "10",
        '"hash # inside quotes"': "hash # inside quotes",
        "'single # quoted'": "single # quoted",
        "abc#nospace": "abc#nospace",
        "  padded  ": "padded",
        "": "",
        "a b c   # words then comment": "a b c",
        "/path/to/x  # a path": "/path/to/x",
        "0.6": "0.6",
    }
    for raw, want in cases.items():
        assert C.parse_value(raw) == want, raw


def test_site_file_never_overrides_the_caller():
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write("AS_TEST_A=file_value   # comment\nexport AS_TEST_B=from_file\n"
                "# comment line\nnot a kv line\n1BAD=skipped\n")
        path = f.name
    os.environ["AS_TEST_A"] = "caller_value"
    os.environ.pop("AS_TEST_B", None)
    try:
        seen = C.load_site_file(path)
        assert os.environ["AS_TEST_A"] == "caller_value", "caller wins"
        assert os.environ["AS_TEST_B"] == "from_file"
        assert seen["AS_TEST_A"] == "file_value"
        assert "1BAD" not in seen
    finally:
        os.environ.pop("AS_TEST_A", None)
        os.environ.pop("AS_TEST_B", None)
        os.unlink(path)


def test_defaults_are_portable():
    """No default may name a directory that exists on only one machine. The repo-derived
    defaults all live under the clone; the two exceptions are a public model id and
    alfworld's own cache convention."""
    for fn in (C.workspace, C.exp_root, C.ckpt_root, C.log_dir):
        old = {k: os.environ.pop(k, None) for k in
               ("ARM_WORKSPACE", "ARM_EXP_ROOT", "ARM_CKPT_ROOT", "ARM_LOG_DIR")}
        try:
            v = fn()
            assert v.startswith(C.repo_root()) or v.startswith("/dev/shm"), (fn.__name__, v)
        finally:
            for k, x in old.items():
                if x is not None:
                    os.environ[k] = x
    assert "/" in C.model_path() and not C.model_path().startswith("/") \
        or os.environ.get("ARM_MODEL"), "default model is a hub id, not a local path"


def test_stamped_paths_isolate_launches():
    a = C.stamped("/x/train.log")
    assert a != "/x/train.log" and a.endswith(".log") and C.run_id() in a


def test_container_memory_returns_numbers_or_none():
    used, limit = C.container_memory()
    if used is not None:
        assert used >= 0 and limit > 0


def test_python_entry_points_carry_the_vllm_toolchain_guards():
    """env.sh and config.py must produce the same serve environment. The A/B serves
    vLLM from the orchestrator process; without these it died JIT-compiling on a
    missing ninja binary — twice, once per resolver, which is why the guard now lives
    in both with the same caller-wins precedence."""
    import subprocess
    import sys
    code = ("import os, json; os.environ.pop('VLLM_ATTENTION_BACKEND', None); "
            "import autoscaffold.config as C; "
            "print(json.dumps({'attn': os.environ.get('VLLM_ATTENTION_BACKEND'), "
            "'smp': os.environ.get('VLLM_USE_FLASHINFER_SAMPLER'), "
            "'path_has_bin': os.path.dirname(os.path.abspath(__import__('sys').executable)) "
            "in os.environ.get('PATH','').split(':')}))")
    import json
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("VLLM_ATTENTION_BACKEND", "VLLM_USE_FLASHINFER_SAMPLER",
                        "ARM_VLLM_ATTN")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, timeout=60)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["attn"] == "FLASH_ATTN"
    assert got["smp"] == "0"
    assert got["path_has_bin"] is True
