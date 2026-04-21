from dataclasses import dataclass


@dataclass
class SuperblockCandidate:
    offset: int
    score: int
    block_size: int


@dataclass
class FSMatch:
    fs_type: str
    offset: int
    confidence: float
