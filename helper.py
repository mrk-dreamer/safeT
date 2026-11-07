from dataclasses import dataclass, field
import numpy as np
import pyqtgraph as pg


@dataclass
class SentenceTimestamp:
    sentence:str
    start:float|None
    end:float|None

    def __post_init__(self):
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("Start time must be less than end time.")

@dataclass
class WordTimestamp:
    word: str
    start: float | None
    end: float | None

    def __post_init__(self):
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("Start time must be less than end time.")

class FormantTarget:
    def __init__(self, timestamp: float, targets: np.ndarray = np.array([]), target_line: pg.InfiniteLine | None = None):
        self.timestamp = timestamp
        self.targets = targets
        self.target_line = target_line

@dataclass
class PhonemeTimestamp:
    phoneme: str
    start: float
    end: float
    parent_sentence_timestamp: SentenceTimestamp
    parent_word_timestamp: WordTimestamp
    formant_targets: list[FormantTarget] = field(default_factory=list)

    def __post_init__(self):
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("Start time must be less than end time.")

def get_phoneme_hash(phoneme_timestamp: PhonemeTimestamp):
        phoneme_hash = id(phoneme_timestamp)
        return phoneme_hash

def ensure_utf8_display(text):
    """Ensure text is properly encoded for display in Qt widgets."""
    if isinstance(text, str):
        # If it's already a string, ensure it's properly encoded
        try:
            # Try to encode and decode to catch any encoding issues
            return text.encode('utf-8', errors='replace').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    elif isinstance(text, bytes):
        # If it's bytes, decode it as UTF-8
        try:
            return text.decode('utf-8', errors='replace')
        except UnicodeDecodeError:
            return str(text, errors='replace')
    return str(text)