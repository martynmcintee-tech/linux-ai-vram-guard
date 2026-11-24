# Linux AI Governor & VRAM Guard (`linux-ai-vram-guard`)

<p align="center">
  <img src="./assets/preview.svg" width="960" alt="Linux AI Governor and VRAM Guard overview" />
</p>

A small, stdlib-only Python 3.10+ daemon I wrote for my 24GB RTX 3090 Ubuntu desktop. It bumps CPU `cpufreq` governors up to `performance` while AI coding CLI tools are running, and automatically unloads idle models from `llama.cpp` (`llama-server` router mode), `ComfyUI`, and `Ollama` so they stop hogging VRAM when you aren't using them.

## Why I wrote this

Running local LLMs, `ComfyUI`, Whisper, and normal desktop apps on a single 24GB GPU kept running into two annoying habits:

1. **Idle models quietly hold onto CUDA memory.** In `llama-server` router mode, even with `--sleep-idle-seconds 300`, presets marked `"sleeping"` can still hang onto residual CUDA context and backend buffers. `ComfyUI` is worse: once an image or video workflow finishes, it leaves checkpoints, text encoders, VAEs, and PyTorch's CUDA cache sitting in VRAM forever (`12 - 20 GiB` with an empty queue). Five minutes later you try to spin up a 32B GGUF or Whisper job and immediately hit `CUDA out of memory`.
2. **CPU governors lag short subprocess bursts.** CLI coding tools (`claude`, `codex`, `antigravity`, `hermes`, `aider`, `cursor`) fire off constant short bursts of `rg`, `git`, AST parses, and test runs. Leaving cores on `powersave` / `schedutil` adds ramp-up lag on bursty work, while pinning `performance` permanently wastes idle power overnight.

## What `ai-vram-guard` does

- Scans `/proc` directly in Python (without forking `ps` every 5 seconds) for active AI coding sessions. When one starts, it writes `performance` to `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`, then drops back to `powersave` (on `intel_pstate` / `amd-pstate-epp`) or `schedutil` (on `acpi-cpufreq`) once they exit.
- Checks `GET /models` on your `llama-server` router (`http://127.0.0.1:18436` by default, with optional `LLAMA_API_KEY` Bearer auth). Whenever an inactive model switches to `"sleeping"`, it calls `POST /models/unload` so the GPU driver gets all of that VRAM back. Pair `llama-server` with `--load-mode dio` on an NVMe SSD and cold reloads only take a couple of seconds on the next prompt.
- Uses `ctypes` against `libnvidia-ml.so.1` and `/proc/net/tcp{,6}` socket inodes to check how much VRAM the `ComfyUI` PID is actually holding. If `ComfyUI` is sitting on `>= 1.5 GiB` and `GET /queue` stays empty for 5 minutes (`300s`), it posts `{"unload_models": true, "free_memory": true}` to `POST /free` to drop the weights and flush `torch.cuda.empty_cache()`.
- Optionally watches `Ollama` (`--ollama-url http://127.0.0.1:11434`) via `GET /api/ps` and evicts idle models (`keep_alive: 0`) after `--idle-seconds`.
- Uses only the Python standard library (no `pip` packages or virtualenv needed, under `15 MB` RSS, and `< 2 ms` per 5-second poll).

## Install

Requires **Linux** (`x86_64` or `aarch64`) with **Python 3.10+** and the NVIDIA driver (`libnvidia-ml.so.1` for per-process VRAM checks; still runs CPU governor switching on non-NVIDIA machines).

### Option 1: `.deb` package (Ubuntu / Debian / Mint / Pop!_OS)

Download `linux-ai-vram-guard_1.0.0_all.deb` from [Releases](https://github.com/martynmcintee-tech/linux-ai-vram-guard/releases/latest) and enable the service:

```bash
sudo apt install ./linux-ai-vram-guard_1.0.0_all.deb
sudo systemctl enable --now ai-vram-guard.service
```

### Option 2: From source (system service: CPU governor + VRAM guard)

```bash
git clone https://github.com/martynmcintee-tech/linux-ai-vram-guard.git
cd linux-ai-vram-guard
chmod +x install.sh
sudo ./install.sh
```

This installs `/usr/local/bin/ai-vram-guard`, `/etc/systemd/system/ai-vram-guard.service`, and `/etc/default/ai-vram-guard` (`0600` root-only config file for overrides and `LLAMA_API_KEY`).

### Option 3: User service (VRAM guard only, no `sudo`)

If you only want idle GPU VRAM cleanup and don't need root `sysfs` governor writes:

```bash
./install.sh --user
```

## Checking live status

```bash
# Plain-text status summary
ai-vram-guard --status

# JSON output
ai-vram-guard --status --json

# Dry-run one check cycle without changing governors or unloading anything
ai-vram-guard --dry-run --once
```

Example output on my desktop:

```text
Linux AI Governor & VRAM Guard v1.0.0
========================================================
CPU Scaling   : driver=intel_pstate | threads=20 | current=performance (active->performance, idle->powersave)
AI Sessions   : ACTIVE (claude[pid=14092], antigravity[24 procs, root_pid=17266])
GPU [0]       : NVIDIA GeForce RTX 3090 - 1.84 / 24.00 GiB VRAM used
CUDA Procs    : Xorg[1842]=0.42GiB, antigravity[17411]=0.38GiB
llama-server  : UP (http://127.0.0.1:18436) - unloaded: qwen3-coder-30b, deepseek-r1-32b
ComfyUI       : IDLE (http://127.0.0.1:8188) - VRAM=0.31 GiB (threshold=1.50 GiB, timeout=300s)
```

## Pairing with `llama-server` router mode

Start your `llama.cpp` router with `--sleep-idle-seconds 300`:

```bash
llama-server \
  --models-preset /path/to/presets.ini \
  --sleep-idle-seconds 300 \
  --host 127.0.0.1 --port 18436
```

After 5 minutes without a prompt, `llama-server` marks the loaded model `"sleeping"`, and on the next poll `ai-vram-guard` calls `POST /models/unload` so its VRAM usage drops all the way to `0.00 GiB`.

If your `llama-server` uses `--api-key`, put `LLAMA_API_KEY=<your_token>` in `/etc/default/ai-vram-guard` and restart:

```bash
sudo systemctl restart ai-vram-guard.service
journalctl -u ai-vram-guard.service -f
```

## CLI flags & environment variables

| CLI Flag | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--llama-url URL` | `LLAMA_URL` | `http://127.0.0.1:18436` | `llama-server` base URL (`off` to disable) |
| `--llama-api-key KEY` | `LLAMA_API_KEY` | *(empty)* | Optional Bearer token (header only sent when set) |
| `--comfy-url URL` | `COMFY_URL` | `http://127.0.0.1:8188` | `ComfyUI` base URL (`off` to disable) |
| `--ollama-url URL` | `OLLAMA_URL` | *(empty)* | Optional `Ollama` base URL (e.g. `http://127.0.0.1:11434`) |
| `--idle-seconds SEC` | `AI_GUARD_IDLE_SECONDS` | `300` | Seconds of empty queue before freeing `ComfyUI` / `Ollama` VRAM |
| `--min-vram-gib GIB` | `AI_GUARD_MIN_VRAM_GIB` | `1.5` | Minimum `ComfyUI` VRAM (GiB) before calling `/free` |
| `--poll-seconds SEC` | `AI_GUARD_POLL_SECONDS` | `5` | Poll interval in seconds |
| `--watch-procs LIST` | `AI_GUARD_WATCH_PROCS` | `claude,codex,antigravity,hermes,aider,cursor` | Comma-separated process names that trigger `performance` mode |
| `--active-governor GOV` | `AI_GUARD_ACTIVE_GOV` | `performance` | Governor used while a watched AI tool is running |
| `--idle-governor GOV` | `AI_GUARD_IDLE_GOV` | `auto` | Idle governor (`auto` picks `powersave` on `intel_pstate`/`amd-pstate-epp`, or `schedutil` on `acpi-cpufreq`) |
| `--no-governor` | `AI_GUARD_NO_GOVERNOR` | `false` | Skip CPU governor switching (VRAM cleanup only) |
| `--gpu-index IDX` | `AI_GUARD_GPU_INDEX` | `-1` | NVML GPU index (`0`, `1`, ...) or `-1` for all NVIDIA GPUs |
| `--once` | *(none)* | `false` | Run one check cycle and exit |
| `--dry-run` | *(none)* | `false` | Log what would happen without writing `sysfs` or unloading models |
| `--status` | *(none)* | `false` | Print current CPU, session, NVML VRAM, and server status, then exit |
| `--json` | *(none)* | `false` | Output JSON when combined with `--status` |

## Uninstall

```bash
sudo ./install.sh --uninstall
# Or for a user-mode install:
./install.sh --uninstall
```

## Notes & Licence

Written for my own Ubuntu 24.04 + RTX 3090 workstation with AI coding tools helping speed up the `ctypes` NVML bindings and `/proc` socket-matching boilerplate.

Released under the [MIT License](LICENSE). Copyright (c) 2026 Martyn McIntee.
