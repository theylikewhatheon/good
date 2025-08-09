"""Enums and constants for the Nexto bot."""

from enum import Enum, auto


class BotMode(Enum):
    """Bot operating modes."""
    NORMAL = auto()
    DEMO = auto()


class DemolishPhase(Enum):
    """State machine for the advanced demo logic."""
    CHOOSE_TARGET = auto()
    CHASING = auto()
    FAILED = auto()