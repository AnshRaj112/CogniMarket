"""OpenEnv/Gym registration helpers for ComputeBazaarEnv."""

from __future__ import annotations

from typing import Any, Dict, Optional


def register_compute_bazaar_env(
    env_id: str = "ComputeBazaar-v0",
    kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Register ComputeBazaarEnv with OpenEnv (if available) and Gymnasium."""
    env_kwargs = kwargs or {}

    try:
        import openenv  # type: ignore

        openenv.register(
            id=env_id,
            entry_point="compute_bazaar_env:ComputeBazaarEnv",
            kwargs=env_kwargs,
        )
    except Exception:
        pass

    try:
        from gymnasium.envs.registration import register

        register(
            id=env_id,
            entry_point="compute_bazaar_env:ComputeBazaarEnv",
            kwargs=env_kwargs,
        )
    except Exception:
        pass

