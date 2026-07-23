"""
simulation/__init__.py
──────────────────────
Marks this directory as a Python package so that
`uvicorn simulation.app:app` resolves correctly.

Public re-exports let other modules import cleanly:
    from simulation import NexusOmniEngine, SimulationParams
"""

from simulation.core_engine import (
    NexusOmniEngine,
    SimulationParams,
    SimulationState,
)

__all__ = ["NexusOmniEngine", "SimulationParams", "SimulationState"]
