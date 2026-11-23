#!/usr/bin/env python3
"""
Linux AI Governor & VRAM Guard (`ai_vram_guard.py`)

Lightweight, zero-dependency Linux daemon for local AI workstations that:
1. Boosts all CPU cores (`intel_pstate`, `amd-pstate-epp`, `acpi-cpufreq`) to
   `performance` while interactive AI coding agents (Claude Code, Codex,
   Antigravity, Hermes, Aider, Cursor, etc.) are running, and reverts cleanly
   to `powersave` / `schedutil` when sessions exit.
2. Monitors `llama.cpp` (`llama-server` router mode) and immediately evicts
   `sleeping` models via `POST /models/unload` so zero residual CUDA context
   VRAM remains pinned on the GPU.
3. Monitors `ComfyUI` GPU memory via direct `libnvidia-ml.so.1` (NVML `ctypes`
   FFI) + `/proc` socket ownership, and posts `POST /free` (`unload_models` +
   `free_memory`) when `/queue` stays idle for `--idle-seconds` (default 300s)
   while holding >= `--min-vram-gib` (default 1.5 GiB) of GPU VRAM.
4. Optionally monitors `Ollama` (`--ollama-url`) via `GET /api/ps` and unloads
   idle resident models after `--idle-seconds`.

License: MIT
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

__version__ = "1.0.0"

NVML_VALUE_NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF
DEFAULT_WATCH_PROCS = "claude,codex,antigravity,hermes,aider,cursor"
SELF_IDENTIFIERS = ("ai_vram_guard", "ai-vram-guard", "ai_governor_vram_guard")
IGNORED_SEARCH_COMMS = frozenset(
    {"grep", "egrep", "fgrep", "rg", "ag", "sed", "awk", "ps", "pgrep", "pkill"}
)
INTERPRETER_PREFIXES = (
    "python",
    "node",
    "bun",
    "deno",
    "uv",
    "npx",
    "tsx",
    "bash",
    "sh",
    "zsh",
    "env",
)


class _NvmlProcessInfoV2(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


class _NvmlProcessInfoV1(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
    ]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


@dataclass
class GpuSummary:
    index: int
    name: str
    used_bytes: int
    total_bytes: int


class NVMLMonitor:
    """Zero-dependency ctypes wrapper around libnvidia-ml.so.1."""

    def __init__(self, gpu_index: int = -1) -> None:
        self.lib: ctypes.CDLL | None = None
        self.handles: list[tuple[int, ctypes.c_void_p, str]] = []
        self._init_nvml(gpu_index)

    def _init_nvml(self, gpu_index: int) -> None:
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                return
            count = ctypes.c_uint(0)
            if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0 or count.value == 0:
                return

            indices = (
                [gpu_index]
                if 0 <= gpu_index < count.value
                else list(range(count.value))
            )
            handles: list[tuple[int, ctypes.c_void_p, str]] = []
            for idx in indices:
                dev = ctypes.c_void_p()
                if lib.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(dev)) == 0:
                    name_buf = ctypes.create_string_buffer(96)
                    gpu_name = f"GPU {idx}"
                    try:
                        if lib.nvmlDeviceGetName(dev, name_buf, 96) == 0:
                            decoded = name_buf.value.decode("utf-8", "replace").strip()
                            if decoded:
                                gpu_name = decoded
                    except Exception:
                        pass
                    handles.append((idx, dev, gpu_name))

            if handles:
                self.lib = lib
                self.handles = handles
        except Exception:
            self.lib = None
            self.handles = []

    def close(self) -> None:
        if self.lib is not None:
            try:
                self.lib.nvmlShutdown()
            except Exception:
                pass
            self.lib = None

    @property
    def available(self) -> bool:
        return self.lib is not None and bool(self.handles)

    @property
    def primary_gpu_label(self) -> str:
        if not self.handles:
            return "GPU"
        if len(self.handles) == 1:
            return self.handles[0][2]
        return f"{self.handles[0][2]} (+{len(self.handles) - 1} more)"

    def get_gpu_summaries(self) -> list[GpuSummary]:
        summaries: list[GpuSummary] = []
        if not self.lib:
            return summaries
        for idx, dev, name in self.handles:
            mem = _NvmlMemory()
            try:
                if self.lib.nvmlDeviceGetMemoryInfo(dev, ctypes.byref(mem)) == 0:
                    summaries.append(
                        GpuSummary(
                            index=idx,
                            name=name,
                            used_bytes=int(mem.used),
                            total_bytes=int(mem.total),
                        )
                    )
            except Exception:
                continue
        return summaries

    def _query_device_procs(self, dev: ctypes.c_void_p) -> dict[int, int]:
        proc_vram: dict[int, int] = {}
        if not self.lib:
            return proc_vram

        # Prefer v3/v2 (MIG-aware struct), fallback to legacy v1 if needed.
        for fn_name, struct_cls in (
            ("nvmlDeviceGetComputeRunningProcesses_v3", _NvmlProcessInfoV2),
            ("nvmlDeviceGetComputeRunningProcesses_v2", _NvmlProcessInfoV2),
            ("nvmlDeviceGetComputeRunningProcesses", _NvmlProcessInfoV1),
        ):
            fn = getattr(self.lib, fn_name, None)
            if fn is None:
                continue
            try:
                count = ctypes.c_uint(0)
                fn(dev, ctypes.byref(count), None)
                if count.value == 0:
                    return proc_vram
                cap = count.value + 16
                arr = (struct_cls * cap)()
                c2 = ctypes.c_uint(cap)
                rc = fn(dev, ctypes.byref(c2), arr)
                if rc == 0:
                    for i in range(c2.value):
                        pid = int(arr[i].pid)
                        mem_b = int(arr[i].usedGpuMemory)
                        if pid > 0 and 0 < mem_b < NVML_VALUE_NOT_AVAILABLE:
                            proc_vram[pid] = proc_vram.get(pid, 0) + mem_b
                    return proc_vram
            except Exception:
                continue
        return proc_vram

    def get_compute_processes(self) -> dict[int, int]:
        totals: dict[int, int] = {}
        for _, dev, _ in self.handles:
            for pid, mem_b in self._query_device_procs(dev).items():
                totals[pid] = totals.get(pid, 0) + mem_b
        return totals


def _normalize_url(raw: str | None) -> str:
    if not raw:
        return ""
    cleaned = raw.strip().rstrip("/")
    if cleaned.lower() in {"none", "off", "false", "0", "disable", "disabled"}:
        return ""
    return cleaned


def _extract_port(url: str, default_port: int) -> int:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.port:
            return int(parsed.port)
        if parsed.scheme == "https":
            return 443
    except Exception:
        pass
    return default_port


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _read_proc_argv(pid: int) -> list[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().decode("utf-8", "replace")
        return [part for part in raw.split("\x00") if part]
    except OSError:
        return []


def _get_port_socket_targets(port: int) -> set[str]:
    target_hex = f":{port:04X}"
    targets: set[str] = set()
    for net_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(net_path, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    parts = line.split()
                    if (
                        len(parts) >= 10
                        and parts[3] == "0A"
                        and parts[1].upper().endswith(target_hex)
                    ):
                        targets.add(f"socket:[{parts[9]}]")
        except OSError:
            continue
    return targets


def _pid_owns_socket(pid: int, socket_targets: set[str]) -> bool:
    if not socket_targets:
        return False
    fd_dir = f"/proc/{pid}/fd"
    try:
        for fd in os.listdir(fd_dir):
            try:
                if os.readlink(f"{fd_dir}/{fd}") in socket_targets:
                    return True
            except OSError:
                continue
    except OSError:
        return False
    return False


def get_comfy_vram_gib(nvml_mon: NVMLMonitor, comfy_port: int) -> float:
    """Measure GPU VRAM (GiB) held by the ComfyUI process."""
    proc_vram = nvml_mon.get_compute_processes()
    if not proc_vram:
        return 0.0
    sock_targets = _get_port_socket_targets(comfy_port)
    total_bytes = 0
    port_str = str(comfy_port)
    for pid, mem_b in proc_vram.items():
        if _pid_owns_socket(pid, sock_targets):
            total_bytes += mem_b
            continue
        cmd = " ".join(_read_proc_argv(pid)).lower()
        if "comfyui" in cmd and ("main.py" in cmd or port_str in cmd):
            total_bytes += mem_b
    return total_bytes / (1024**3)


def _token_matches_segment(token: str, candidate: str) -> bool:
    """Match a watched token against a process name or path without substring noise."""
    cand = candidate.lower()
    if cand == token:
        return True
    parts = [p for p in re.split(r"[/\\:\s@._-]+", cand) if p]
    return token in parts


def find_active_ai_agents(watch_tokens: list[str]) -> list[tuple[int, str]]:
    """Scan /proc directly for running AI agent processes matching watch_tokens."""
    if not watch_tokens:
        return []
    self_pid = os.getpid()
    parent_pid = os.getppid()
    matches: list[tuple[int, str]] = []

    try:
        entries = os.listdir("/proc")
    except OSError:
        return matches

    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == self_pid:
            continue

        comm = _read_text(f"/proc/{pid}/comm").lower()
        if not comm or comm in IGNORED_SEARCH_COMMS:
            continue

        argv = _read_proc_argv(pid)
        joined_lower = " ".join(argv).lower()
        if any(ident in joined_lower or ident in comm for ident in SELF_IDENTIFIERS):
            continue
        if pid == parent_pid and "--watch-procs" in joined_lower:
            continue

        # Candidate strings: kernel comm, executable path argv[0], and script entrypoints
        # when launched via an interpreter (node, python, uv, bash, etc.).
        candidates: list[str] = [comm]
        if argv:
            exe = argv[0]
            candidates.append(exe)
            exe_base = os.path.basename(exe).lower()
            if exe_base.startswith(INTERPRETER_PREFIXES) or comm.startswith(
                INTERPRETER_PREFIXES
            ):
                for arg in argv[1:5]:
                    if not arg or arg == "-m":
                        continue
                    if arg.startswith("-"):
                        continue
                    candidates.append(arg)
                    break

        matched_token: str | None = None
        for token in watch_tokens:
            if any(_token_matches_segment(token, c) for c in candidates):
                matched_token = token
                break

        if matched_token is not None:
            matches.append((pid, matched_token))

    matches.sort(key=lambda item: item[0])
    return matches


class CpuGovernorController:
    """Manages Linux cpufreq scaling_governor across Intel and AMD systems."""

    def __init__(
        self,
        active_governor: str = "performance",
        idle_governor: str = "auto",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.governor_paths = self._discover_governor_paths()
        self.driver = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver")
        avail_raw = _read_text(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors"
        )
        self.available_governors = avail_raw.split() if avail_raw else []
        self.active_governor = active_governor
        self.idle_governor = self._resolve_idle_governor(idle_governor)
        self._warned_perm = False

    @staticmethod
    def _discover_governor_paths() -> list[str]:
        cpu_root = "/sys/devices/system/cpu"
        paths: list[tuple[int, str]] = []
        try:
            for entry in os.listdir(cpu_root):
                m = re.fullmatch(r"cpu(\d+)", entry)
                if not m:
                    continue
                gov_path = os.path.join(cpu_root, entry, "cpufreq", "scaling_governor")
                if os.path.isfile(gov_path):
                    paths.append((int(m.group(1)), gov_path))
        except OSError:
            return []
        paths.sort(key=lambda item: item[0])
        return [p for _, p in paths]

    def _resolve_idle_governor(self, requested: str) -> str:
        if requested and requested.lower() != "auto":
            return requested
        # On intel_pstate and amd-pstate-epp (active), 'powersave' is the hardware-guided
        # dynamic frequency state. On acpi-cpufreq / passive pstate, 'schedutil' scales
        # dynamically whereas 'powersave' pins minimum frequency.
        if self.driver in {"intel_pstate", "amd-pstate-epp", "amd_pstate_epp"}:
            if not self.available_governors or "powersave" in self.available_governors:
                return "powersave"
        if "schedutil" in self.available_governors:
            return "schedutil"
        if "powersave" in self.available_governors:
            return "powersave"
        if "ondemand" in self.available_governors:
            return "ondemand"
        return "powersave"

    def current_governors(self) -> set[str]:
        vals: set[str] = set()
        for path in self.governor_paths:
            val = _read_text(path)
            if val:
                vals.add(val)
        return vals

    def primary_governor(self) -> str:
        if not self.governor_paths:
            return "unavailable"
        return _read_text(self.governor_paths[0]) or "unknown"

    def apply(
        self,
        target: str,
        reason: str,
        dry_run: bool = False,
    ) -> None:
        if not self.enabled or not self.governor_paths:
            return
        current = self.current_governors()
        if current == {target}:
            return

        cur_desc = "/".join(sorted(current)) if current else "unknown"
        drv_desc = self.driver or "cpufreq"
        n_cpus = len(self.governor_paths)

        if dry_run:
            print(
                f"[AI-Governor] [DRY-RUN] Would switch {n_cpus} CPU threads ({drv_desc}) "
                f"from '{cur_desc}' -> '{target}' ({reason})",
                flush=True,
            )
            return

        wrote = 0
        perm_denied = False
        for path in self.governor_paths:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(target)
                wrote += 1
            except PermissionError:
                perm_denied = True
                break
            except OSError:
                continue

        if perm_denied:
            if not self._warned_perm:
                print(
                    "[AI-Governor] Permission denied writing sysfs scaling_governor "
                    "(run as root via systemd or pass --no-governor); continuing in VRAM-Guard mode.",
                    flush=True,
                )
                self._warned_perm = True
            self.enabled = False
            return

        if wrote > 0:
            print(
                f"[AI-Governor] Switched {wrote} CPU threads ({drv_desc}) "
                f"from '{cur_desc}' -> '{target}' ({reason})",
                flush=True,
            )


def _llama_headers(api_key: str, json_body: bool = False) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def query_llama_models(
    llama_url: str, api_key: str
) -> tuple[list[dict[str, Any]] | None, str]:
    """Return (models_list_or_None, state_label: 'ok'|'disabled'|'auth_error'|'unreachable')."""
    if not llama_url:
        return None, "disabled"
    req = urllib.request.Request(
        f"{llama_url}/models",
        headers=_llama_headers(api_key),
    )
    try:
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data", [])
        return (data if isinstance(data, list) else []), "ok"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return None, f"auth_error:{exc.code}"
        return None, f"http_error:{exc.code}"
    except Exception:
        return None, "unreachable"


_WARNED_LLAMA_AUTH = False


def unload_sleeping_llama_models(
    llama_url: str,
    api_key: str,
    gpu_label: str,
    dry_run: bool = False,
) -> int:
    """When llama-server marks a model 'sleeping', unload it completely from GPU VRAM."""
    global _WARNED_LLAMA_AUTH
    models, state_label = query_llama_models(llama_url, api_key)
    if state_label.startswith("auth_error"):
        if not _WARNED_LLAMA_AUTH:
            code = state_label.split(":", 1)[-1]
            print(
                f"[VRAM-Guard] llama-server at {llama_url} returned HTTP {code}; "
                "set LLAMA_API_KEY (or --llama-api-key) to enable sleeping-model unload.",
                flush=True,
            )
            _WARNED_LLAMA_AUTH = True
        return 0
    if not models:
        return 0

    unloaded = 0
    for model in models:
        mid = model.get("id")
        raw_status = model.get("status", {})
        state = (
            raw_status.get("value", "")
            if isinstance(raw_status, dict)
            else str(raw_status)
        )
        if not mid or state != "sleeping":
            continue

        if dry_run:
            print(
                f"[VRAM-Guard] [DRY-RUN] Would unload sleeping llama-server model '{mid}' from {gpu_label} VRAM.",
                flush=True,
            )
            unloaded += 1
            continue

        body = json.dumps({"model": mid}).encode("utf-8")
        ureq = urllib.request.Request(
            f"{llama_url}/models/unload",
            data=body,
            headers=_llama_headers(api_key, json_body=True),
            method="POST",
        )
        try:
            with urllib.request.urlopen(ureq, timeout=5.0) as uresp:
                if uresp.status == 200:
                    print(
                        f"[VRAM-Guard] Unloaded sleeping llama-server model '{mid}' from {gpu_label} VRAM.",
                        flush=True,
                    )
                    unloaded += 1
        except Exception:
            continue
    return unloaded


def query_comfy_queue(comfy_url: str) -> dict[str, Any] | None:
    if not comfy_url:
        return None
    try:
        q_req = urllib.request.Request(f"{comfy_url}/queue")
        with urllib.request.urlopen(q_req, timeout=2.0) as q_resp:
            data = json.loads(q_resp.read().decode("utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def guard_comfy_idle_vram(
    nvml_mon: NVMLMonitor,
    comfy_url: str,
    comfy_port: int,
    min_vram_gib: float,
    idle_limit_secs: float,
    idle_since: float | None,
    dry_run: bool = False,
) -> float | None:
    """If ComfyUI holds >= min_vram_gib with an empty queue for >= idle_limit_secs, POST /free."""
    if not comfy_url:
        return None

    vram_gib = get_comfy_vram_gib(nvml_mon, comfy_port)
    if vram_gib < min_vram_gib:
        return None

    q_data = query_comfy_queue(comfy_url)
    if q_data is None:
        return None

    busy = bool(q_data.get("queue_running")) or bool(q_data.get("queue_pending"))
    if busy:
        return None

    now = time.monotonic()
    if idle_since is None:
        return now

    if (now - idle_since) >= idle_limit_secs:
        if dry_run:
            print(
                f"[VRAM-Guard] [DRY-RUN] Would POST {comfy_url}/free to reclaim "
                f"{vram_gib:.2f} GiB ComfyUI VRAM on {nvml_mon.primary_gpu_label} "
                f"after {idle_limit_secs:.0f}s idle.",
                flush=True,
            )
            return now

        payload = json.dumps({"unload_models": True, "free_memory": True}).encode(
            "utf-8"
        )
        freq = urllib.request.Request(
            f"{comfy_url}/free",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(freq, timeout=5.0) as fresp:
                if fresp.status == 200:
                    print(
                        f"[VRAM-Guard] Freed idle ComfyUI VRAM ({vram_gib:.2f} GiB on "
                        f"{nvml_mon.primary_gpu_label}) after {idle_limit_secs:.0f}s inactivity.",
                        flush=True,
                    )
        except Exception:
            pass
        return now

    return idle_since


def query_ollama_ps(ollama_url: str) -> list[dict[str, Any]] | None:
    if not ollama_url:
        return None
    try:
        req = urllib.request.Request(f"{ollama_url}/api/ps")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("models", [])
        return models if isinstance(models, list) else []
    except Exception:
        return None


def guard_ollama_idle_vram(
    ollama_url: str,
    idle_limit_secs: float,
    tracker: dict[str, tuple[str, float]],
    gpu_label: str,
    dry_run: bool = False,
) -> None:
    """Optional Ollama VRAM guard: unload resident models untouched for >= idle_limit_secs."""
    if not ollama_url:
        tracker.clear()
        return

    models = query_ollama_ps(ollama_url)
    if models is None:
        return

    now = time.monotonic()
    seen: set[str] = set()
    for m in models:
        name = m.get("name") or m.get("model")
        size_vram = int(m.get("size_vram") or 0)
        expires_at = str(m.get("expires_at") or "")
        if not name or size_vram <= 0:
            continue
        seen.add(name)
        prev = tracker.get(name)
        # Whenever Ollama serves a new request for a model, its expires_at lease shifts forward.
        if prev is None or prev[0] != expires_at:
            tracker[name] = (expires_at, now)
            continue

        if (now - prev[1]) >= idle_limit_secs:
            vram_gib = size_vram / (1024**3)
            if dry_run:
                print(
                    f"[VRAM-Guard] [DRY-RUN] Would unload idle Ollama model '{name}' "
                    f"({vram_gib:.2f} GiB on {gpu_label}) after {idle_limit_secs:.0f}s inactivity.",
                    flush=True,
                )
                tracker[name] = (expires_at, now)
                continue

            body = json.dumps({"model": name, "keep_alive": 0}).encode("utf-8")
            req = urllib.request.Request(
                f"{ollama_url}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if resp.status == 200:
                        print(
                            f"[VRAM-Guard] Unloaded idle Ollama model '{name}' "
                            f"({vram_gib:.2f} GiB on {gpu_label}) after {idle_limit_secs:.0f}s inactivity.",
                            flush=True,
                        )
            except Exception:
                pass
            tracker.pop(name, None)

    for stale in list(tracker.keys()):
        if stale not in seen:
            tracker.pop(stale, None)


def _summarize_active_agents(agents: list[tuple[int, str]]) -> str:
    grouped: dict[str, list[int]] = {}
    for pid, tok in agents:
        grouped.setdefault(tok, []).append(pid)
    parts: list[str] = []
    for tok, pids in grouped.items():
        if len(pids) == 1:
            parts.append(f"{tok}[pid={pids[0]}]")
        else:
            parts.append(f"{tok}[{len(pids)} procs, root_pid={pids[0]}]")
    return ", ".join(parts)


def collect_status(
    args: argparse.Namespace,
    gov_ctrl: CpuGovernorController,
    nvml_mon: NVMLMonitor,
    watch_tokens: list[str],
    comfy_port: int,
) -> dict[str, Any]:
    agents = find_active_ai_agents(watch_tokens)
    gpus = nvml_mon.get_gpu_summaries()
    proc_vram = nvml_mon.get_compute_processes()

    llama_models, llama_state = query_llama_models(args.llama_url, args.llama_api_key)
    comfy_vram_gib = (
        get_comfy_vram_gib(nvml_mon, comfy_port) if args.comfy_url else 0.0
    )
    comfy_queue = query_comfy_queue(args.comfy_url) if args.comfy_url else None
    ollama_models = query_ollama_ps(args.ollama_url) if args.ollama_url else None

    return {
        "version": __version__,
        "cpu": {
            "governor_management_enabled": gov_ctrl.enabled,
            "driver": gov_ctrl.driver or "unavailable",
            "threads": len(gov_ctrl.governor_paths),
            "current_governors": sorted(gov_ctrl.current_governors()),
            "active_target": gov_ctrl.active_governor,
            "idle_target": gov_ctrl.idle_governor,
        },
        "watched_agents": {
            "patterns": watch_tokens,
            "active": [{"pid": pid, "match": tok} for pid, tok in agents],
        },
        "gpus": [
            {
                "index": g.index,
                "name": g.name,
                "used_gib": round(g.used_bytes / (1024**3), 2),
                "total_gib": round(g.total_bytes / (1024**3), 2),
            }
            for g in gpus
        ],
        "compute_processes": [
            {
                "pid": pid,
                "comm": _read_text(f"/proc/{pid}/comm") or "unknown",
                "vram_gib": round(mem_b / (1024**3), 2),
            }
            for pid, mem_b in sorted(
                proc_vram.items(), key=lambda kv: kv[1], reverse=True
            )
        ],
        "llama_server": {
            "url": args.llama_url or None,
            "state": llama_state,
            "reachable": llama_state == "ok" or llama_state.startswith("auth_error"),
            "auth_configured": bool(args.llama_api_key),
            "models": [
                {
                    "id": m.get("id"),
                    "state": (
                        m.get("status", {}).get("value", "unknown")
                        if isinstance(m.get("status"), dict)
                        else str(m.get("status", "unknown"))
                    ),
                }
                for m in (llama_models or [])
            ],
        },
        "comfyui": {
            "url": args.comfy_url or None,
            "reachable": comfy_queue is not None,
            "vram_gib": round(comfy_vram_gib, 2),
            "min_vram_gib": args.min_vram_gib,
            "idle_seconds": args.idle_seconds,
            "busy": (
                bool(comfy_queue.get("queue_running"))
                or bool(comfy_queue.get("queue_pending"))
                if comfy_queue is not None
                else False
            ),
        },
        "ollama": {
            "url": args.ollama_url or None,
            "reachable": ollama_models is not None if args.ollama_url else False,
            "models": [
                {
                    "name": m.get("name") or m.get("model"),
                    "vram_gib": round(int(m.get("size_vram") or 0) / (1024**3), 2),
                }
                for m in (ollama_models or [])
            ],
        },
    }


def print_status_report(status: dict[str, Any]) -> None:
    cpu = status["cpu"]
    govs = "/".join(cpu["current_governors"]) if cpu["current_governors"] else "n/a"
    print(f"Linux AI Governor & VRAM Guard v{status['version']}")
    print("=" * 56)
    print(
        f"CPU Scaling   : driver={cpu['driver']} | threads={cpu['threads']} | "
        f"current={govs} (active->{cpu['active_target']}, idle->{cpu['idle_target']})"
    )

    active_agents = [
        (int(a["pid"]), str(a["match"])) for a in status["watched_agents"]["active"]
    ]
    if active_agents:
        print(f"AI Sessions   : ACTIVE ({_summarize_active_agents(active_agents)})")
    else:
        print(
            f"AI Sessions   : idle (watching: {', '.join(status['watched_agents']['patterns'])})"
        )

    if status["gpus"]:
        for g in status["gpus"]:
            print(
                f"GPU [{g['index']}]       : {g['name']} — "
                f"{g['used_gib']:.2f} / {g['total_gib']:.2f} GiB VRAM used"
            )
    else:
        print("GPU           : NVML unavailable (no NVIDIA GPU detected)")

    if status["compute_processes"]:
        procs = ", ".join(
            f"{p['comm']}[{p['pid']}]={p['vram_gib']:.2f}GiB"
            for p in status["compute_processes"]
        )
        print(f"CUDA Procs    : {procs}")

    ls = status["llama_server"]
    if ls["url"]:
        if ls["state"] == "ok":
            by_state: dict[str, list[str]] = {}
            for m in ls["models"]:
                by_state.setdefault(m["state"], []).append(str(m["id"]))
            summary = (
                "; ".join(f"{st}: {', '.join(ids)}" for st, ids in by_state.items())
                if by_state
                else "0 models registered"
            )
            print(f"llama-server  : UP ({ls['url']}) — {summary}")
        elif str(ls["state"]).startswith("auth_error"):
            code = str(ls["state"]).split(":", 1)[-1]
            print(
                f"llama-server  : AUTH REQUIRED (HTTP {code} at {ls['url']} — set LLAMA_API_KEY)"
            )
        else:
            print(f"llama-server  : unreachable ({ls['url']})")
    else:
        print("llama-server  : disabled")

    cf = status["comfyui"]
    if cf["url"]:
        state_str = (
            "BUSY" if cf["busy"] else ("IDLE" if cf["reachable"] else "unreachable")
        )
        print(
            f"ComfyUI       : {state_str} ({cf['url']}) — "
            f"VRAM={cf['vram_gib']:.2f} GiB (threshold={cf['min_vram_gib']:.2f} GiB, "
            f"timeout={cf['idle_seconds']:.0f}s)"
        )
    else:
        print("ComfyUI       : disabled")

    ol = status["ollama"]
    if ol["url"]:
        if ol["reachable"]:
            m_desc = (
                ", ".join(f"{m['name']} ({m['vram_gib']:.2f} GiB)" for m in ol["models"])
                or "0 resident models"
            )
            print(f"Ollama        : UP ({ol['url']}) — {m_desc}")
        else:
            print(f"Ollama        : unreachable ({ol['url']})")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-vram-guard",
        description=(
            "Boost Linux CPU scaling governors during active AI coding sessions "
            "and automatically reclaim idle GPU VRAM from llama-server, ComfyUI, and Ollama."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--llama-url",
        default=os.environ.get("LLAMA_URL", "http://127.0.0.1:18436"),
        help="llama.cpp router base URL (env: LLAMA_URL; pass 'off' to disable).",
    )
    parser.add_argument(
        "--llama-api-key",
        default=os.environ.get("LLAMA_API_KEY", ""),
        help="Optional Bearer token for llama-server (env: LLAMA_API_KEY; omitted when empty).",
    )
    parser.add_argument(
        "--comfy-url",
        default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"),
        help="ComfyUI base URL (env: COMFY_URL; pass 'off' to disable).",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", ""),
        help="Optional Ollama base URL, e.g. http://127.0.0.1:11434 (env: OLLAMA_URL).",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=float(os.environ.get("AI_GUARD_IDLE_SECONDS", "300")),
        help="Seconds of queue inactivity before freeing ComfyUI / Ollama VRAM (env: AI_GUARD_IDLE_SECONDS).",
    )
    parser.add_argument(
        "--min-vram-gib",
        type=float,
        default=float(os.environ.get("AI_GUARD_MIN_VRAM_GIB", "1.5")),
        help="Minimum GPU VRAM in GiB held by ComfyUI before triggering /free (env: AI_GUARD_MIN_VRAM_GIB).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.environ.get("AI_GUARD_POLL_SECONDS", "5")),
        help="Daemon polling interval in seconds (env: AI_GUARD_POLL_SECONDS).",
    )
    parser.add_argument(
        "--watch-procs",
        default=os.environ.get("AI_GUARD_WATCH_PROCS", DEFAULT_WATCH_PROCS),
        help="Comma-separated AI agent process names to watch (env: AI_GUARD_WATCH_PROCS).",
    )
    parser.add_argument(
        "--active-governor",
        default=os.environ.get("AI_GUARD_ACTIVE_GOV", "performance"),
        help="CPU cpufreq governor while an AI agent is running (env: AI_GUARD_ACTIVE_GOV).",
    )
    parser.add_argument(
        "--idle-governor",
        default=os.environ.get("AI_GUARD_IDLE_GOV", "auto"),
        help="CPU cpufreq governor when idle ('auto' selects powersave or schedutil; env: AI_GUARD_IDLE_GOV).",
    )
    parser.add_argument(
        "--no-governor",
        action="store_true",
        default=os.environ.get("AI_GUARD_NO_GOVERNOR", "").lower()
        in {"1", "true", "yes"},
        help="Disable CPU scaling_governor management (VRAM Guard only; env: AI_GUARD_NO_GOVERNOR=1).",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=int(os.environ.get("AI_GUARD_GPU_INDEX", "-1")),
        help="NVML GPU index to monitor, or -1 for all NVIDIA GPUs (env: AI_GUARD_GPU_INDEX).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute a single check cycle and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log governor and VRAM actions without mutating sysfs or unloading models.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current CPU governor, watched agents, GPU VRAM, and server status, then exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output when combined with --status.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    args.llama_url = _normalize_url(args.llama_url)
    args.comfy_url = _normalize_url(args.comfy_url)
    args.ollama_url = _normalize_url(args.ollama_url)
    args.llama_api_key = (args.llama_api_key or "").strip()

    watch_tokens = [
        tok.strip().lower() for tok in args.watch_procs.split(",") if tok.strip()
    ]
    comfy_port = _extract_port(args.comfy_url, 8188) if args.comfy_url else 8188

    gov_ctrl = CpuGovernorController(
        active_governor=args.active_governor,
        idle_governor=args.idle_governor,
        enabled=not args.no_governor,
    )
    nvml_mon = NVMLMonitor(gpu_index=args.gpu_index)

    stop_requested = False

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        if args.status:
            status = collect_status(args, gov_ctrl, nvml_mon, watch_tokens, comfy_port)
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print_status_report(status)
            return 0

        if not args.once:
            mode_tag = " [DRY-RUN]" if args.dry_run else ""
            print(
                f"[AI-Governor & VRAM-Guard] v{__version__} daemon started{mode_tag} | "
                f"GPU={nvml_mon.primary_gpu_label} | CPU={gov_ctrl.driver or 'n/a'} "
                f"({len(gov_ctrl.governor_paths)} threads) | idle={args.idle_seconds:.0f}s",
                flush=True,
            )

        comfy_idle_since: float | None = None
        ollama_tracker: dict[str, tuple[str, float]] = {}

        while not stop_requested:
            try:
                active_agents = find_active_ai_agents(watch_tokens)
                if active_agents:
                    summary = _summarize_active_agents(active_agents)
                    gov_ctrl.apply(
                        gov_ctrl.active_governor,
                        reason=f"active: {summary}",
                        dry_run=args.dry_run,
                    )
                else:
                    gov_ctrl.apply(
                        gov_ctrl.idle_governor,
                        reason="no watched AI agents running",
                        dry_run=args.dry_run,
                    )

                if args.llama_url:
                    unload_sleeping_llama_models(
                        llama_url=args.llama_url,
                        api_key=args.llama_api_key,
                        gpu_label=nvml_mon.primary_gpu_label,
                        dry_run=args.dry_run,
                    )

                if args.comfy_url:
                    comfy_idle_since = guard_comfy_idle_vram(
                        nvml_mon=nvml_mon,
                        comfy_url=args.comfy_url,
                        comfy_port=comfy_port,
                        min_vram_gib=args.min_vram_gib,
                        idle_limit_secs=args.idle_seconds,
                        idle_since=comfy_idle_since,
                        dry_run=args.dry_run,
                    )

                if args.ollama_url:
                    guard_ollama_idle_vram(
                        ollama_url=args.ollama_url,
                        idle_limit_secs=args.idle_seconds,
                        tracker=ollama_tracker,
                        gpu_label=nvml_mon.primary_gpu_label,
                        dry_run=args.dry_run,
                    )
            except Exception as exc:
                print(f"[AI-Governor & VRAM-Guard] Cycle error: {exc}", flush=True)

            if args.once or stop_requested:
                break

            # Sleep in small increments so SIGTERM stops systemd cleanly within 0.25s.
            deadline = time.monotonic() + max(0.5, args.poll_seconds)
            while not stop_requested and time.monotonic() < deadline:
                time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))

        if not args.once:
            print("[AI-Governor & VRAM-Guard] Daemon stopped cleanly.", flush=True)
        return 0
    finally:
        nvml_mon.close()


if __name__ == "__main__":
    sys.exit(main())
