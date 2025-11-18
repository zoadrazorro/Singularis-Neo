"""
AURA-Brain Module

Biological neural simulation with neuromodulation.
"""

from .bio_simulator import (
    AURABrainSimulator,
    NeuromodulatorType,
    SpikingNeuron,
    SynapticConnection,
)

__all__ = [
    'AURABrainSimulator',
    'NeuromodulatorType',
    'SpikingNeuron',
    'SynapticConnection',
]
