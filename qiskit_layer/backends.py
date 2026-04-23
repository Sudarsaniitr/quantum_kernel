"""
qiskit_layer/backends.py
========================
Backend helpers for Aer simulator and IBM Quantum hardware.

Handles two execution targets:

  AerSimulator  — local simulator shipped with qiskit-aer.
                  Optionally accepts a NoiseModel for depolarising error injection.

  IBM hardware  — accessed via qiskit-ibm-runtime's QiskitRuntimeService.
                  Requires a token in QISKIT_IBM_TOKEN (see .env.example).
                  Backend names: ibm_kingston, ibm_marrakesh, ibm_fez (open-plan).

Channel migration note
----------------------
IBM Quantum changed its channel name from "ibm_quantum" to "ibm_quantum_platform"
in late 2024.  _normalize_runtime_channel() maps the old name to the new one so
that existing .env files and legacy configs keep working without changes.

Token validation
----------------
_normalize_runtime_token() rejects empty strings and placeholder values like
"YOUR_API_TOKEN_HERE" so that code running without real credentials falls
through to a saved-account fallback instead of raising a confusing 401 error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _load_dotenv_if_available(env_file: Optional[str] = None) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_file, override=False)
    except Exception:
        # dotenv is optional; environment variables may already be present.
        pass


@dataclass
class IBMRuntimeConfig:
    token: Optional[str]
    instance: Optional[str]
    channel: str


def _normalize_runtime_channel(channel: Optional[str]) -> str:
    """
    Normalize legacy/new channel names for qiskit-ibm-runtime.

    Newer runtime versions accept:
      - "ibm_quantum_platform"
      - "ibm_cloud"

    We keep backward compatibility with legacy "ibm_quantum" by mapping it
    to "ibm_quantum_platform".
    """
    c = (channel or "").strip().lower()
    if c in {"", "ibm_quantum", "ibm_quantum_platform"}:
        return "ibm_quantum_platform"
    if c == "ibm_cloud":
        return "ibm_cloud"
    return c


def _normalize_runtime_token(token: Optional[str]) -> Optional[str]:
    """Return None for empty/placeholder tokens; otherwise return stripped token."""
    t = (token or "").strip()
    if not t:
        return None

    tu = t.upper()
    if "YOUR_" in tu or "API_TOKEN_HERE" in tu:
        return None

    return t


def get_ibm_runtime_config(
    token: Optional[str] = None,
    instance: Optional[str] = None,
    channel: str = "ibm_quantum_platform",
    env_file: Optional[str] = None,
) -> IBMRuntimeConfig:
    """Get runtime config from args/env."""
    _load_dotenv_if_available(env_file)

    raw_channel = os.getenv("QISKIT_IBM_CHANNEL", channel)

    return IBMRuntimeConfig(
        token=_normalize_runtime_token(token or os.getenv("QISKIT_IBM_TOKEN")),
        instance=instance or os.getenv("QISKIT_IBM_INSTANCE"),
        channel=_normalize_runtime_channel(raw_channel),
    )


def backend_name(backend) -> str:
    """Compatibility helper for backend naming across API versions."""
    n = getattr(backend, "name", None)
    if callable(n):
        try:
            return str(n())
        except Exception:
            pass
    if isinstance(n, str):
        return n
    return backend.__class__.__name__


def get_aer_simulator(noise_model=None, seed_simulator: int = 1234):
    """Return configured Aer simulator backend."""
    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "qiskit-aer is required for simulator mode. "
            "Install with: pip install -r requirements-qiskit.txt"
        ) from exc

    kwargs = {"seed_simulator": seed_simulator}
    if noise_model is not None:
        kwargs["noise_model"] = noise_model
    return AerSimulator(**kwargs)


def get_runtime_service(config: IBMRuntimeConfig):
    """Create IBM runtime service using token (or saved account if present)."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "qiskit-ibm-runtime is required for IBM backend mode. "
            "Install with: pip install -r requirements-qiskit.txt"
        ) from exc

    if config.token:
        return QiskitRuntimeService(
            channel=config.channel,
            token=config.token,
            instance=config.instance,
        )

    # fallback to pre-saved account
    return QiskitRuntimeService(channel=config.channel, instance=config.instance)


def get_ibm_backend(backend_name_value: str, config: IBMRuntimeConfig):
    """Resolve an IBM backend by name."""
    if not backend_name_value:
        raise ValueError("backend_name_value must be provided for IBM mode.")

    service = get_runtime_service(config)
    return service.backend(backend_name_value)
