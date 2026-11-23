#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./install.sh [OPTIONS]

Install Linux AI Governor & VRAM Guard (`ai-vram-guard`).

Options:
  (default)     System-wide install to /usr/local/bin/ai-vram-guard +
                /etc/systemd/system/ai-vram-guard.service (enables CPU cpufreq
                governor switching + NVML GPU VRAM reclamation). Requires root/sudo.
  --user        Rootless install to ~/.local/bin/ai-vram-guard +
                ~/.config/systemd/user/ai-vram-guard.service (--no-governor
                VRAM-Guard-only mode).
  --uninstall   Stop, disable, and remove the installed service and binary.
  -h, --help    Show this help message.
EOF
}

MODE="system"
for arg in "$@"; do
  case "$arg" in
    --user) MODE="user" ;;
    --uninstall) MODE="uninstall" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "$MODE" == "uninstall" ]]; then
  if [[ "${EUID}" -eq 0 ]]; then
    systemctl disable --now ai-vram-guard.service 2>/dev/null || true
    rm -f /etc/systemd/system/ai-vram-guard.service /usr/local/bin/ai-vram-guard
    systemctl daemon-reload
    echo "Removed system service ai-vram-guard.service and /usr/local/bin/ai-vram-guard"
    echo "(Kept /etc/default/ai-vram-guard if present; remove manually if desired.)"
  else
    systemctl --user disable --now ai-vram-guard.service 2>/dev/null || true
    rm -f "${HOME}/.config/systemd/user/ai-vram-guard.service" "${HOME}/.local/bin/ai-vram-guard"
    systemctl --user daemon-reload
    echo "Removed user service ai-vram-guard.service and ~/.local/bin/ai-vram-guard"
  fi
  exit 0
fi

if [[ "$MODE" == "user" ]]; then
  BIN_DIR="${HOME}/.local/bin"
  UNIT_DIR="${HOME}/.config/systemd/user"
  CONF_DIR="${HOME}/.config/ai-vram-guard"
  mkdir -p "${BIN_DIR}" "${UNIT_DIR}" "${CONF_DIR}"

  install -m 0755 "${SCRIPT_DIR}/ai_vram_guard.py" "${BIN_DIR}/ai-vram-guard"

  if [[ ! -f "${CONF_DIR}/env" ]]; then
    cat > "${CONF_DIR}/env" <<'EOF'
# Optional overrides for ai-vram-guard (chmod 600)
# LLAMA_URL=http://127.0.0.1:18436
# LLAMA_API_KEY=
# COMFY_URL=http://127.0.0.1:8188
# OLLAMA_URL=
# AI_GUARD_IDLE_SECONDS=300
# AI_GUARD_MIN_VRAM_GIB=1.5
EOF
    chmod 0600 "${CONF_DIR}/env"
  fi

  cat > "${UNIT_DIR}/ai-vram-guard.service" <<EOF
[Unit]
Description=Linux AI VRAM Auto-Unload Guard (User Mode)
After=default.target

[Service]
Type=simple
EnvironmentFile=-%h/.config/ai-vram-guard/env
ExecStart=/usr/bin/python3 %h/.local/bin/ai-vram-guard --no-governor
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

  systemctl --user daemon-reload
  systemctl --user enable --now ai-vram-guard.service
  echo "Installed and started user service ai-vram-guard.service (VRAM-Guard mode)."
  echo "Check status with: ${BIN_DIR}/ai-vram-guard --status"
  exit 0
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "System-wide installation requires root (for sysfs cpufreq governor access)." >&2
  echo "Run: sudo ./install.sh   (or './install.sh --user' for rootless VRAM-only mode)" >&2
  exit 1
fi

install -m 0755 "${SCRIPT_DIR}/ai_vram_guard.py" /usr/local/bin/ai-vram-guard
install -m 0644 "${SCRIPT_DIR}/systemd/ai-vram-guard.service" /etc/systemd/system/ai-vram-guard.service

if [[ ! -f /etc/default/ai-vram-guard ]]; then
  cat > /etc/default/ai-vram-guard <<'EOF'
# Configuration for ai-vram-guard.service (root-readable 0600)
# Uncomment to customize endpoints, Bearer API key, idle timeout, or watched agents:
#
# LLAMA_URL=http://127.0.0.1:18436
# LLAMA_API_KEY=
# COMFY_URL=http://127.0.0.1:8188
# OLLAMA_URL=
# AI_GUARD_IDLE_SECONDS=300
# AI_GUARD_MIN_VRAM_GIB=1.5
# AI_GUARD_POLL_SECONDS=5
# AI_GUARD_WATCH_PROCS=claude,codex,antigravity,hermes,aider,cursor
# AI_GUARD_ACTIVE_GOV=performance
# AI_GUARD_IDLE_GOV=auto
EOF
  chmod 0600 /etc/default/ai-vram-guard
fi

systemctl daemon-reload
systemctl enable --now ai-vram-guard.service

echo "Installed and started ai-vram-guard.service!"
echo "Configure optional settings in /etc/default/ai-vram-guard"
echo "Inspect live telemetry anytime with: ai-vram-guard --status"
