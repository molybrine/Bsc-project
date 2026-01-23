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

    def _do_tense(self, v, t):
        return (v[:-2] + 'is') if t == 'past' else (v[:-2] + 'os') if t == 'future' else v

    def _eng_v(self, ev, t):
        self.call_count += 1
        if t == 'past':
            return _PAST.get(ev, ev + 'ed')
        elif t == 'future':
            return 'will ' + ev.rstrip('s')
        else:
            return ev

    def _case(self, s, o):
        return s + self.NOM, o + self.ACC

    def _build_t(self, s, v, o, t='present'):
        vv = self._do_tense(v, t)
        sn, oa = self._case(s, o)
        return {
            'v1': s + ' ' + vv + ' ' + o,
            'v2': vv + ' ' + s + ' ' + o,
            'v3': sn + ' ' + vv + ' ' + oa,
            'v4': vv + ' ' + sn + ' ' + oa,
        }

    def _build_i(self, s, v, t='present'):
        vv = self._do_tense(v, t)
        sn = s + self.NOM
        r = {}
        r['v1'] = f'{s} {vv}'
        r['v2'] = f'{vv} {s}'
        r['v3'] = f'{sn} {vv}'
        r['v4'] = f'{vv} {sn}'
        return r

    def _build_d(self, s, v, io, do, t='present'):
        vv = self._do_tense(v, t)
        sn, doa = self._case(s, do)
        return dict(
            v1=f'{s} {vv} {io} {do}',
            v2=f'{vv} {s} {io} {do}',
            v3=f'{sn} {vv} {io} {doa}',
            v4=f'{vv} {sn} {io} {doa}',
        )

    def generate_test_set(self, nt=20, ni=20, nd=20, na=10):
        _init_words()
        ss = []
        nn = list(WORDS['n'].keys())
        vv = list(WORDS['v'].keys())
        tt = ['past', 'present', 'future']

        for i in range(nt):
            _s = self.rng.choice(nn)
            _v = self.rng.choice(vv)
            _o = self.rng.choice([x for x in nn if x != _s])
            _t = tt[i % 3]
            _vars = self._build_t(f'la {_s}', _v, f'la {_o}', _t)
            _es = WORDS['n'][_s]
            _ev = self._eng_v(WORDS['v'][_v], _t)
            _eo = WORDS['n'][_o]
            ss.append(S(
                english=f'the {_es} {_ev} the {_eo}',
                verb_type='transitive', has_adjective=False,
                has_pronoun=False, tense=_t,
                word_count=len(_vars['v1'].split()),
                **_vars,
            ))

        _iv = {'mangxas': 'eats', 'trinkas': 'drinks', 'legas': 'reads', 'skribas': 'writes'}
        _seen = set()
        _i = 0
        while _i < ni:
            _s = self.rng.choice(nn)
            _v = self.rng.choice(list(_iv.keys()))
            _t = tt[_i % 3]
            _vars = self._build_i(f'la {_s}', _v, _t)
            if _vars['v1'] in _seen:
                continue
            _seen.add(_vars['v1'])
            _es = WORDS['n'][_s]
            _ev = self._eng_v(_iv[_v], _t)
            ss.append(S(
                english=f'the {_es} {_ev}',
                verb_type='intransitive', has_adjective=False,
                has_pronoun=False, tense=_t,
                word_count=len(_vars['v1'].split()),
                **_vars,
            ))
            _i += 1

        _dv = ['donas', 'portas', 'metas']
        for i in range(nd):
            _s = self.rng.choice(nn)
            _v = self.rng.choice(_dv)
            _io = self.rng.choice([x for x in nn if x != _s])
            _do = self.rng.choice([x for x in nn if x not in (_s, _io)])
            _t = tt[i % 3]
            _vars = self._build_d(f'la {_s}', _v, f'al la {_io}', f'la {_do}', _t)
            _es = WORDS['n'][_s]
            _ev = self._eng_v(WORDS['v'][_v], _t)
            _eio = WORDS['n'][_io]
            _edo = WORDS['n'][_do]
            ss.append(S(
                english=f'the {_es} {_ev} to the {_eio} the {_edo}',
                verb_type='ditransitive', has_adjective=False,
                has_pronoun=False, tense=_t,
                word_count=len(_vars['v1'].split()),
                **_vars,
            ))

        _aa = list(WORDS['a'].keys())
        for i in range(na):
            _s = self.rng.choice(nn)
            _adj = self.rng.choice(_aa)
            _v = self.rng.choice(vv)
            _o = self.rng.choice([x for x in nn if x != _s])
            _t = tt[i % 3]
            _vars = self._build_t(f'la {_adj} {_s}', _v, f'la {_o}', _t)
            _es = WORDS['n'][_s]
            _ea = WORDS['a'][_adj]
            _ev = self._eng_v(WORDS['v'][_v], _t)
            _eo = WORDS['n'][_o]
            ss.append(S(
                english=f'the {_ea} {_es} {_ev} the {_eo}',
                verb_type='transitive', has_adjective=True,
                has_pronoun=False, tense=_t,
                word_count=len(_vars['v1'].split()),
                **_vars,
            ))

        return ss

