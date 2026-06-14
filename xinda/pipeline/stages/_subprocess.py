"""Shared subprocess runner for LaTeXML invocations (live stdout + timeout)."""

from __future__ import annotations

import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from xinda.logger_config import setup_logger

logger = setup_logger(__name__)


def run_command_live(
    cmd: list[str], log_file_path: Path | str, timeout: int = 1800
) -> bool:
    """Run a command, streaming stdout to a log file. Returns success bool.

    Watches for LaTeXML's `Status:conversion:0|1` sentinel to detect success
    even when the process exits with a nonzero code (LaTeXML often does this
    on benign warnings).
    """
    log_file_path = Path(log_file_path)
    success = False
    with open(log_file_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"command: {' '.join(cmd)}\n")
        log_f.write(f"start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"timeout: {timeout}s\n\n")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        stop_event = threading.Event()

        def monitor():
            t0 = time.time()
            while time.time() - t0 < timeout:
                if stop_event.is_set():
                    return
                time.sleep(1)
            if process.poll() is None:
                process.kill()
                log_f.write(f"\n!!! timeout {timeout}s; process killed\n")
                logger.error("subprocess timeout: %s", cmd[0])

        t = threading.Thread(target=monitor, daemon=True)
        t.start()

        try:
            for line in iter(process.stdout.readline, ""):
                log_f.write(line)
                if "Status:conversion:0" in line or "Status:conversion:1" in line:
                    success = True
                    break
            process.wait()
            if process.returncode != 0 and not success:
                log_f.write(f"\n!!! returncode {process.returncode}\n")
                logger.error("subprocess failed: %s rc=%d", cmd[0], process.returncode)
            else:
                log_f.write(f"\nreturncode {process.returncode} (success)\n")
                success = True
        finally:
            stop_event.set()
            t.join(1)
            if process.poll() is None:
                process.kill()
            log_f.write(f"end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return success
