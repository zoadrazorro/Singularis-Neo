"""
Singularis Integration Wrappers
================================

Integration adapters for connecting DATA system with existing Singularis components.
"""

from .data_consciousness_bridge import DATAConsciousnessBridge
from .data_lifeops_bridge import DATALifeOpsBridge  
from .data_skyrim_bridge import DATASkyrimBridge

__all__ = [
    "DATAConsciousnessBridge",
    "DATALifeOpsBridge",
    "DATASkyrimBridge",
]

