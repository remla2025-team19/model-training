import pytest

SYNONYM_PAIRS = [
    ('okay', 'fine'),
    ('okay', 'ok'),
    ('good', 'nice'),
    ('bad', 'terrible'),
    ('bad', 'not ok'),
    ('delicious', 'tasty'),
]

@pytest.mark.parametrize('a,b', SYNONYM_PAIRS)
@pytest.mark.parametrize('orig,alt', [
    ('The food was okay and service good.', 'The food was fine and service nice.'),
    ('A bad experience overall.', 'An awful experience overall.'),
])

def test_synonym_invariance(model, orig, alt):
    assert model(orig) == model(alt)

def test_single_word_invariance(model, a, b):
    orig = f'The {a} pizza'
    alt  = f'The {b} pizza'
    assert model(orig) == model(alt)
