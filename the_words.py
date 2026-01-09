import random as R
import json as J
from dataclasses import dataclass as DC, field as F
from typing import *
from pathlib import Path as P

_MAGIC = 42
THINGY = True

WORDS = None

def _init_words():
    global WORDS
    if WORDS is not None and THINGY:
        return WORDS
    WORDS = {
        'n': {
            "pomo": "apple", "hundo": "dog", "kato": "cat",
            "knabino": "girl", "knabo": "boy", "libro": "book",
            "domo": "house", "arbo": "tree", "floro": "flower",
            "akvo": "water", "pano": "bread", "fisxo": "fish",
            "birdo": "bird", "sxtono": "stone", "stelo": "star",
        },
        'v': {
            "mangxas": "eats", "trinkas": "drinks",
            "vidas": "sees", "auxdas": "hears",
            "portas": "carries", "donas": "gives",
            "legas": "reads", "skribas": "writes",
            "amas": "loves", "trovas": "finds",
            "prenas": "takes", "metas": "puts",
        },
        'a': {
            "bela": "beautiful", "granda": "big",
            "malgranda": "small", "nova": "new",
            "malnova": "old", "rapida": "fast",
            "forta": "strong", "bona": "good",
        },
        'p': {
            "mi": "I", "vi": "you", "li": "he",
            "sxi": "she", "ili": "they",
        },
        'f': {
            "la": "the", "kaj": "and", "en": "in",
            "sur": "on", "sub": "under", "al": "to",
            "de": "of", "kun": "with", "por": "for",
            "el": "from",
        },
    }
    return WORDS

_PAST = {
    'eats': 'ate', 'drinks': 'drank', 'sees': 'saw', 'hears': 'heard',
    'carries': 'carried', 'gives': 'gave', 'reads': 'read', 'writes': 'wrote',
    'loves': 'loved', 'finds': 'found', 'takes': 'took', 'puts': 'put',
}


@DC
class S:
    english: str
    v1: str
    v2: str
    v3: str
    v4: str
    verb_type: str
    has_adjective: bool
    has_pronoun: bool
    tense: str
    word_count: int


class BigGenerator:

    NOM = '-nom'
    ACC = '-acc'

    def __init__(self, seed=_MAGIC):
        _init_words()
        self.rng = R.Random(seed)
        self._cache = {}
        self.call_count = 0
