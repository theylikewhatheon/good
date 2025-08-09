"""
Nexto bot package - A modular Rocket League bot.

This package provides a well-structured Rocket League bot with:
- Separated concerns (physics, demo logic, kickoffs)
- Modular design for easy maintenance
- Clear separation between normal and demo modes
"""

from .nexto import Nexto
from .vector_utils import Vec3
from .bot_enums import BotMode, DemolishPhase

__all__ = ['Nexto', 'Vec3', 'BotMode', 'DemolishPhase']