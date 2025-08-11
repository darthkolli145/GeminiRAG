import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag.utils import split_into_sentences

def test_split_into_sentences_basic():
    text = "Hello world. This is great! Right?"
    assert split_into_sentences(text) == ["Hello world.", "This is great!", "Right?"]


def test_split_into_sentences_max_sentences():
    text = "One. Two. Three. Four."
    assert split_into_sentences(text, max_sentences=2) == ["One.", "Two."]
