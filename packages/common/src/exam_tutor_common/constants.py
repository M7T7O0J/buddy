from __future__ import annotations

from enum import Enum


class TutorMode(str, Enum):
    doubt = "doubt"
    practice = "practice"
    pyq = "pyq"


class Exam(str, Enum):
    GATE_DA = "GATE_DA"
    GATE_CS = "GATE_CS"
    UPSC_PRELIMS = "UPSC_PRELIMS"
    UPSC_MAINS = "UPSC_MAINS"
