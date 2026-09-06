"""Valetudo robot vacuum control."""

import logging

from src.tools.http_utils import http_put

log = logging.getLogger(__name__)

_vacuum_ip: str = ""


def init_vacuum(cfg: dict) -> None:
    global _vacuum_ip
    _vacuum_ip = cfg.get("tools", {}).get("vacuum", {}).get("ip", "")
    if _vacuum_ip:
        log.info(f"Vacuum configured at {_vacuum_ip}")
    else:
        log.warning("Vacuum IP not configured — start_vacuum will return an error")


def start_vacuum() -> str:
    if not _vacuum_ip:
        return (
            "ERROR: Vacuum IP not configured."
            " Tell the user the vacuum is unavailable."
        )
    try:
        http_put(
            f"http://{_vacuum_ip}/api/v2/robot/capabilities/BasicControlCapability",
            json={"action": "start"},
            timeout=5,
        )
        log.info("Vacuum cleaning started")
        return "Starting to clean."
    except Exception as e:
        log.error(f"Vacuum start failed: {e}")
        return (
            f"ERROR: Could not start the vacuum: {e}."
            " Tell the user the command failed."
        )
