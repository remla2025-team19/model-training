import pytest
import re

# Synonym mapping for tests
SYNONYM_PAIRS = [
    ('okay', 'fine'),
    ('good', 'nice'),
    ('delicious', 'tasty'),
]

def _swap(text: str, a: str, b: str) -> str:
    """Replace word a with b in text, preserving word boundaries and case-insensitive."""
    return re.sub(rf"\b{re.escape(a)}\b", b, text, flags=re.IGNORECASE)

@pytest.mark.parametrize('orig,a,b', [
    ('The food was okay and service good.', 'okay', 'fine'),
])
def test_synonym_invariance(model, orig, a, b):
    """Replacing key words with synonyms should not change sentiment."""
    alt = _swap(orig, a, b)
    assert model(orig) == model(alt)

@pytest.mark.parametrize('a,b', SYNONYM_PAIRS)
def test_single_word_invariance(model, a, b):
    """Single-token substitution invariance (e.g. good→nice)."""
    orig = f'The {a} pizza.'
    alt = _swap(orig, a, b)
    assert model(orig) == model(alt)
