"""
Module 1: Synthetic Language Generator
Generates vocabulary and sentences in four grammatical variants:
  V1 (SVO, no case)  -- Baseline
  V2 (VSO, no case)  -- Syntax only
  V3 (SVO, + case)   -- Morphology only
  V4 (VSO, + case)   -- Both features
"""
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path


@dataclass
class Lexicon:
    """Esperanto-inspired vocabulary with 50 items."""
    nouns: Dict[str, str] = field(default_factory=lambda: {
        "pomo": "apple", "hundo": "dog", "kato": "cat",
        "knabino": "girl", "knabo": "boy", "libro": "book",
        "domo": "house", "arbo": "tree", "floro": "flower",
        "akvo": "water", "pano": "bread", "fisxo": "fish",
        "birdo": "bird", "sxtono": "stone", "stelo": "star",
    })
    verbs: Dict[str, str] = field(default_factory=lambda: {
        "mangxas": "eats", "trinkas": "drinks",
        "vidas": "sees", "auxdas": "hears",
        "portas": "carries", "donas": "gives",
        "legas": "reads", "skribas": "writes",
        "amas": "loves", "trovas": "finds",
        "prenas": "takes", "metas": "puts",
    })
    adjectives: Dict[str, str] = field(default_factory=lambda: {
        "bela": "beautiful", "granda": "big",
        "malgranda": "small", "nova": "new",
        "malnova": "old", "rapida": "fast",
        "forta": "strong", "bona": "good",
    })
    pronouns: Dict[str, str] = field(default_factory=lambda: {
        "mi": "I", "vi": "you", "li": "he",
        "sxi": "she", "ili": "they",
    })
    function_words: Dict[str, str] = field(default_factory=lambda: {
        "la": "the", "kaj": "and", "en": "in",
        "sur": "on", "sub": "under", "al": "to",
        "de": "of", "kun": "with", "por": "for",
        "el": "from",
    })


@dataclass
class Sentence:
    """Represents a single test sentence."""
    english: str
    v1: str   # SVO, no case
    v2: str   # VSO, no case
    v3: str   # SVO, with case
    v4: str   # VSO, with case
    verb_type: str        # transitive / intransitive / ditransitive
    has_adjective: bool
    has_pronoun: bool
    tense: str            # past / present / future
    word_count: int


class LanguageGenerator:
    """Generates sentences in all four variants."""

    CASE_NOM = '-nom'
    CASE_ACC = '-acc'

    TENSE_MAP = {
        'past':    lambda v: v[:-2] + 'is',
        'present': lambda v: v,              # already -as
        'future':  lambda v: v[:-2] + 'os',
    }

    # English tense forms for each verb
    ENG_TENSE = {
        'past':    {
            'eats': 'ate', 'drinks': 'drank',
            'sees': 'saw', 'hears': 'heard',
            'carries': 'carried', 'gives': 'gave',
            'reads': 'read', 'writes': 'wrote',
            'loves': 'loved', 'finds': 'found',
            'takes': 'took', 'puts': 'put',
        },
        'present': {},  # use base form as-is
        'future': {},   # prefix with 'will'
    }

    def _english_verb(
        self, eng_verb: str, tense: str
    ) -> str:
        """Apply tense to English verb."""
        if tense == 'past':
            return self.ENG_TENSE['past'].get(
                eng_verb, eng_verb + 'ed'
            )
        elif tense == 'future':
            return f'will {eng_verb.rstrip("s")}'
        return eng_verb  # present

    def __init__(self, seed: int = 42):
        self.lexicon = Lexicon()
        self.rng = random.Random(seed)

    def _apply_tense(self, verb: str, tense: str) -> str:
        return self.TENSE_MAP[tense](verb)

    def _add_case(self, subj: str, obj: str) -> Tuple[str, str]:
        return subj + self.CASE_NOM, obj + self.CASE_ACC

    def _build_transitive(
        self, subj: str, verb: str, obj: str,
        tense: str = 'present'
    ) -> Dict[str, str]:
        v = self._apply_tense(verb, tense)
        s_nom, o_acc = self._add_case(subj, obj)
        return {
            'v1': f'{subj} {v} {obj}',
            'v2': f'{v} {subj} {obj}',
            'v3': f'{s_nom} {v} {o_acc}',
            'v4': f'{v} {s_nom} {o_acc}',
        }

    def _build_intransitive(
        self, subj: str, verb: str, tense: str = 'present'
    ) -> Dict[str, str]:
        v = self._apply_tense(verb, tense)
        s_nom = subj + self.CASE_NOM
        return {
            'v1': f'{subj} {v}',
            'v2': f'{v} {subj}',
            'v3': f'{s_nom} {v}',
            'v4': f'{v} {s_nom}',
        }

    def _build_ditransitive(
        self, subj: str, verb: str, io: str, do: str,
        tense: str = 'present'
    ) -> Dict[str, str]:
        v = self._apply_tense(verb, tense)
        s_nom, do_acc = self._add_case(subj, do)
        return {
            'v1': f'{subj} {v} {io} {do}',
            'v2': f'{v} {subj} {io} {do}',
            'v3': f'{s_nom} {v} {io} {do_acc}',
            'v4': f'{v} {s_nom} {io} {do_acc}',
        }

    def generate_test_set(
        self, n_transitive: int = 20,
        n_intransitive: int = 20,
        n_ditransitive: int = 20,
        n_adjective: int = 10,
    ) -> List[Sentence]:
        """Generate the full 70-sentence test set."""
        sentences = []
        nouns = list(self.lexicon.nouns.keys())
        verbs = list(self.lexicon.verbs.keys())
        tenses = ['past', 'present', 'future']

        # Transitive sentences (20)
        for i in range(n_transitive):
            s = self.rng.choice(nouns)
            v = self.rng.choice(verbs)
            o = self.rng.choice([n for n in nouns if n != s])
            t = tenses[i % 3]
            variants = self._build_transitive(
                f'la {s}', v, f'la {o}', t
            )
            eng_s = self.lexicon.nouns[s]
            eng_v = self._english_verb(
                self.lexicon.verbs[v], t
            )
            eng_o = self.lexicon.nouns[o]
            sentences.append(Sentence(
                english=f'the {eng_s} {eng_v} the {eng_o}',
                verb_type='transitive',
                has_adjective=False,
                has_pronoun=False,
                tense=t,
                word_count=len(variants['v1'].split()),
                **variants
            ))

        # Intransitive sentences (20)
        intrans_verbs = {
            'mangxas': 'eats', 'trinkas': 'drinks',
            'legas': 'reads', 'skribas': 'writes',
        }
        seen_intrans = set()
        i = 0
        while i < n_intransitive:
            s = self.rng.choice(nouns)
            v = self.rng.choice(
                list(intrans_verbs.keys())
            )
            t = tenses[i % 3]
            variants = self._build_intransitive(
                f'la {s}', v, t
            )
            # Skip duplicates
            if variants['v1'] in seen_intrans:
                continue
            seen_intrans.add(variants['v1'])
            eng_s = self.lexicon.nouns[s]
            eng_v = self._english_verb(
                intrans_verbs[v], t
            )
            sentences.append(Sentence(
                english=f'the {eng_s} {eng_v}',
                verb_type='intransitive',
                has_adjective=False,
                has_pronoun=False,
                tense=t,
                word_count=len(variants['v1'].split()),
                **variants
            ))
            i += 1

        # Ditransitive sentences (20)
        ditrans_verbs = ['donas', 'portas', 'metas']
        for i in range(n_ditransitive):
            s = self.rng.choice(nouns)
            v = self.rng.choice(ditrans_verbs)
            io = self.rng.choice(
                [n for n in nouns if n != s]
            )
            do = self.rng.choice(
                [n for n in nouns
                 if n not in (s, io)]
            )
            t = tenses[i % 3]
            variants = self._build_ditransitive(
                f'la {s}', v,
                f'al la {io}', f'la {do}', t
            )
            eng_s = self.lexicon.nouns[s]
            eng_v = self._english_verb(
                self.lexicon.verbs[v], t
            )
            eng_io = self.lexicon.nouns[io]
            eng_do = self.lexicon.nouns[do]
            sentences.append(Sentence(
                english=(f'the {eng_s} {eng_v} '
                         f'to the {eng_io} '
                         f'the {eng_do}'),
                verb_type='ditransitive',
                has_adjective=False,
                has_pronoun=False,
                tense=t,
                word_count=len(variants['v1'].split()),
                **variants
            ))

        # Adjective-modified transitive sentences (10)
        adjs = list(self.lexicon.adjectives.keys())
        for i in range(n_adjective):
            s = self.rng.choice(nouns)
            adj = self.rng.choice(adjs)
            v = self.rng.choice(verbs)
            o = self.rng.choice(
                [n for n in nouns if n != s]
            )
            t = tenses[i % 3]
            variants = self._build_transitive(
                f'la {adj} {s}', v, f'la {o}', t
            )
            eng_s = self.lexicon.nouns[s]
            eng_adj = self.lexicon.adjectives[adj]
            eng_v = self._english_verb(
                self.lexicon.verbs[v], t
            )
            eng_o = self.lexicon.nouns[o]
            sentences.append(Sentence(
                english=(f'the {eng_adj} {eng_s} '
                         f'{eng_v} the {eng_o}'),
                verb_type='transitive',
                has_adjective=True,
                has_pronoun=False,
                tense=t,
                word_count=len(variants['v1'].split()),
                **variants
            ))

        return sentences

    def generate_few_shot_examples(
        self, variant: str, n: int = 8
    ) -> List[Dict[str, str]]:
        """Generate n diverse demonstration pairs
        including transitive, intransitive, and
        ditransitive examples for coverage."""
        examples = []
        nouns = list(self.lexicon.nouns.keys())
        verbs = list(self.lexicon.verbs.keys())
        # 5 transitive, 2 intransitive, 1 ditransitive
        n_trans = min(n, max(n - 3, 1))
        n_intrans = min(2, n - n_trans)
        n_ditrans = n - n_trans - n_intrans

        # Transitive examples
        for i in range(n_trans):
            s = nouns[i % len(nouns)]
            v = verbs[i % len(verbs)]
            o = nouns[(i + 3) % len(nouns)]
            eng = (f'the {self.lexicon.nouns[s]} '
                   f'{self.lexicon.verbs[v]} '
                   f'the {self.lexicon.nouns[o]}')
            variants = self._build_transitive(
                f'la {s}', v, f'la {o}'
            )
            examples.append({
                'english': eng,
                'translation': variants[variant],
            })

        # Intransitive examples
        for i in range(n_intrans):
            s = nouns[(i + 10) % len(nouns)]
            v = verbs[(i + 8) % len(verbs)]
            eng = (f'the {self.lexicon.nouns[s]} '
                   f'{self.lexicon.verbs[v]}')
            variants = self._build_intransitive(
                f'la {s}', v
            )
            examples.append({
                'english': eng,
                'translation': variants[variant],
            })

        # Ditransitive examples
        for i in range(n_ditrans):
            s = nouns[(i + 12) % len(nouns)]
            v = verbs[(i + 10) % len(verbs)]
            io = nouns[(i + 13) % len(nouns)]
            do = nouns[(i + 14) % len(nouns)]
            eng = (f'the {self.lexicon.nouns[s]} '
                   f'{self.lexicon.verbs[v]} '
                   f'to the {self.lexicon.nouns[io]} '
                   f'the {self.lexicon.nouns[do]}')
            variants = self._build_ditransitive(
                f'la {s}', v,
                f'al la {io}', f'la {do}'
            )
            examples.append({
                'english': eng,
                'translation': variants[variant],
            })

        return examples

    def save_dataset(self, path: str = 'data/test_set.json'):
        """Export test set to JSON."""
        sentences = self.generate_test_set()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [vars(s) for s in sentences]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f'Saved {len(data)} sentences to {path}')
        return sentences


if __name__ == '__main__':
    gen = LanguageGenerator(seed=42)
    gen.save_dataset()
