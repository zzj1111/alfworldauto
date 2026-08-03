"""Best-effort Resend email + atomic status-file writer (stdlib only)."""
import json
import os
import urllib.request

SECRETS = "/mnt/data1/zha00175/tool-agent-secrets/resend.env"


def _load():
    env = {}
    try:
        with open(SECRETS) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k] = v
    except Exception:
        pass
    return env


def send_email(subject, text):
    env = _load()
    key = env.get("RESEND_API_KEY") or os.environ.get("RESEND_API_KEY")
    if not key:
        return False, "no_key"
    to = env.get("RESEND_TO", "zha00175@umn.edu")
    frm = env.get("RESEND_FROM", "onboarding@resend.dev")
    body = json.dumps({"from": frm, "to": [to], "subject": subject, "text": text}).encode()
    req = urllib.request.Request(
        "https://api.resend.com/emails", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 # Resend is behind Cloudflare, which 403s (error 1010) the default
                 # python-urllib User-Agent as a bot. A browser-like UA passes.
                 "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
            return ("id" in d), d.get("id", str(d))
    except Exception as e:
        return False, str(e)[:200]


def write_status(path, status):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(status, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        pass
