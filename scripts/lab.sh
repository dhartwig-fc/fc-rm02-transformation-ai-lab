#!/usr/bin/env bash
# Financial Crime AI Lab - local stack helper.
#
#   scripts/lab.sh init     create .env (secret generated; macOS -> native Ollama mode)
#   scripts/lab.sh up       start the stack
#   scripts/lab.sh models   pull the chat + embedding models from .env
#   scripts/lab.sh check    smoke test: services healthy, localhost-only, model answers
#   scripts/lab.sh down     stop (data volumes are kept)
#   scripts/lab.sh logs     follow logs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

die()  { echo "ERROR: $*" >&2; exit 1; }
ok()   { echo "  [ok]   $*"; }
fail() { echo "  [FAIL] $*"; FAILED=1; }

load_env() {
  [[ -f .env ]] || die ".env missing - run: scripts/lab.sh init"
  set -a; # shellcheck disable=SC1091
  source .env; set +a
  OLLAMA_PORT="${OLLAMA_PORT:-11434}"
  OPEN_WEBUI_PORT="${OPEN_WEBUI_PORT:-3000}"
  OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
}

# docker mode = Ollama runs in a container; native = Ollama installed on the host.
mode() { [[ "$OLLAMA_BASE_URL" == http://ollama:* ]] && echo docker || echo native; }

compose() {
  if [[ "$(mode)" == docker ]]; then docker compose --profile docker-ollama "$@"
  else docker compose "$@"; fi
}

ollama_cli() {
  if [[ "$(mode)" == docker ]]; then docker exec fc-ollama ollama "$@"
  else command -v ollama >/dev/null || die "native mode: install Ollama from https://ollama.com"; ollama "$@"; fi
}

cmd_init() {
  if [[ -f .env ]]; then echo ".env already exists - leaving it alone."; return; fi
  cp .env.example .env
  local secret
  secret="$(openssl rand -hex 32 2>/dev/null || python3 -c 'import secrets;print(secrets.token_hex(32))')"
  sed -i.bak "s/^WEBUI_SECRET_KEY=.*/WEBUI_SECRET_KEY=${secret}/" .env
  if [[ "$(uname -s)" == Darwin ]]; then
    sed -i.bak "s#^OLLAMA_BASE_URL=.*#OLLAMA_BASE_URL=http://host.docker.internal:11434#" .env
    echo "macOS detected -> native Ollama mode (uses the Apple GPU). Install Ollama from https://ollama.com"
  fi
  rm -f .env.bak
  echo "Created .env (mode: $(load_env; mode))."
}

cmd_up() {
  load_env
  echo "Starting stack in $(mode) mode..."
  compose up -d
  echo "Open WebUI: http://127.0.0.1:${OPEN_WEBUI_PORT}  (first account created = admin)"
}

cmd_models() {
  load_env
  echo "Pulling ${LAB_CHAT_MODEL} and ${LAB_EMBED_MODEL}..."
  ollama_cli pull "$LAB_CHAT_MODEL"
  ollama_cli pull "$LAB_EMBED_MODEL"
  ollama_cli list
}

cmd_check() {
  load_env
  FAILED=0
  echo "Smoke test ($(mode) mode)"

  # 1. Ports must be bound to loopback only.
  local bad
  bad="$(compose config --format json | python3 -c '
import json, sys
cfg = json.load(sys.stdin)
for name, svc in cfg.get("services", {}).items():
    for p in svc.get("ports", []):
        ip = p.get("host_ip", "") if isinstance(p, dict) else ""
        if ip != "127.0.0.1":
            print(f"{name}:{p}")
')"
  [[ -z "$bad" ]] && ok "all published ports bound to 127.0.0.1" || fail "non-loopback ports: $bad"

  # 2. Ollama reachable.
  if curl -fsS -m 5 "http://127.0.0.1:${OLLAMA_PORT}/api/version" >/dev/null; then
    ok "Ollama responding on 127.0.0.1:${OLLAMA_PORT}"
  else fail "Ollama not reachable on 127.0.0.1:${OLLAMA_PORT}"; fi

  # 3. Open WebUI healthy.
  if curl -fsS -m 5 "http://127.0.0.1:${OPEN_WEBUI_PORT}/health" >/dev/null; then
    ok "Open WebUI healthy on 127.0.0.1:${OPEN_WEBUI_PORT}"
  else fail "Open WebUI not healthy yet (first start can take ~1 min)"; fi

  # 4. Chat model answers (skipped if not pulled yet).
  if curl -fsS -m 5 "http://127.0.0.1:${OLLAMA_PORT}/api/tags" | grep -q "\"${LAB_CHAT_MODEL}\""; then
    local reply
    reply="$(curl -fsS -m 300 "http://127.0.0.1:${OLLAMA_PORT}/api/generate" \
      -d "{\"model\":\"${LAB_CHAT_MODEL}\",\"prompt\":\"Reply with exactly: LAB-OK\",\"stream\":false,\"think\":false}" \
      | python3 -c 'import json,sys;print(json.load(sys.stdin).get("response","").strip())' || true)"
    [[ "$reply" == *LAB-OK* ]] && ok "${LAB_CHAT_MODEL} answered" || fail "${LAB_CHAT_MODEL} reply unexpected: ${reply:0:80}"
  else
    echo "  [skip] ${LAB_CHAT_MODEL} not pulled yet - run: scripts/lab.sh models"
  fi

  [[ "$FAILED" == 0 ]] && echo "All checks passed." || { echo "Some checks failed."; exit 1; }
}

cmd_down() { load_env; compose down; }
cmd_logs() { load_env; compose logs -f; }

case "${1:-}" in
  init) cmd_init ;; up) cmd_up ;; models) cmd_models ;; check) cmd_check ;;
  down) cmd_down ;; logs) cmd_logs ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
