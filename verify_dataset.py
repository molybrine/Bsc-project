"""
Dataset Verification Script
Run after language generation to validate the test set.
Usage:
    python verify_dataset.py
    python verify_dataset.py --path data/test_set.json
    python verify_dataset.py --verbose
"""
import json
import argparse
import sys
from collections import Counter
from pathlib import Path


class DatasetVerifier:
    """Validates the generated test set JSON."""

    def __init__(self, path: str = 'data/test_set.json'):
        self.path = Path(path)
        self.data = []
        self.errors = []
        self.warnings = []

    def load(self) -> bool:
        """Load and parse the JSON file."""
        if not self.path.exists():
            self.errors.append(
                f'File not found: {self.path}'
            )
            return False
        try:
            with open(self.path, 'r') as f:
                self.data = json.load(f)
            print(f'Loaded {len(self.data)} sentences '
                  f'from {self.path}')
            return True
        except json.JSONDecodeError as e:
            self.errors.append(
                f'Invalid JSON: {e}'
            )
            return False

    def check_sentence_count(self):
        """Verify we have exactly 70 sentences."""
        n = len(self.data)
        if n != 70:
            self.errors.append(
                f'Expected 70 sentences, found {n}'
            )
        else:
            print(f'  [PASS] Sentence count: {n}')

    def check_required_fields(self):
        """Ensure every sentence has all fields."""
        required = {
            'english', 'v1', 'v2', 'v3', 'v4',
            'verb_type', 'has_adjective',
            'has_pronoun', 'tense', 'word_count',
        }
        missing = []
        for i, sent in enumerate(self.data):
            diff = required - set(sent.keys())
            if diff:
                missing.append((i, diff))
        if missing:
            self.errors.append(
                f'Missing fields in {len(missing)} '
                f'sentences: {missing[:3]}...'
            )
        else:
            print(f'  [PASS] All required fields present')

    def check_verb_type_distribution(self):
        """Verify sentence type counts."""
        counts = Counter(
            s['verb_type'] for s in self.data
        )
        expected = {
            'transitive': 20,
            'intransitive': 20,
            'ditransitive': 20,
        }
        print(f'  Verb type distribution: '
              f'{dict(counts)}')
        for vtype, exp in expected.items():
            actual = counts.get(vtype, 0)
            if actual != exp:
                self.warnings.append(
                    f'{vtype}: expected {exp}, '
                    f'found {actual}'
                )

    def check_tense_distribution(self):
        """Check tense coverage."""
        counts = Counter(
            s['tense'] for s in self.data
        )
        print(f'  Tense distribution: {dict(counts)}')
        for tense in ('past', 'present', 'future'):
            if counts.get(tense, 0) == 0:
                self.warnings.append(
                    f'No sentences with tense: {tense}'
                )

    def check_duplicates(self):
        """Detect duplicate sentences in any variant."""
        for variant in ('v1', 'v2', 'v3', 'v4'):
            texts = [s[variant] for s in self.data]
            dupes = [
                t for t, c in Counter(texts).items()
                if c > 1
            ]
            if dupes:
                self.errors.append(
                    f'Duplicates in {variant}: '
                    f'{dupes[:3]}'
                )
            else:
                print(f'  [PASS] No duplicates in {variant}')

    def check_variant_structure(self):
        """Validate structural properties of each variant."""
        verb_suffixes = ('as', 'is', 'os')
        errors = []
        for i, sent in enumerate(self.data):
            v1_tokens = sent['v1'].split()
            v2_tokens = sent['v2'].split()
            v3_tokens = sent['v3'].split()
            v4_tokens = sent['v4'].split()

            # V1: SVO - verb should NOT be first
            if len(v1_tokens) > 1:
                if any(v1_tokens[0].endswith(s)
                       for s in verb_suffixes):
                    errors.append(
                        f'Sent {i}: V1 looks VSO '
                        f'("{sent["v1"]}")'
                    )

            # V2: VSO - verb SHOULD be first
            if len(v2_tokens) > 1:
                if not any(v2_tokens[0].endswith(s)
                           for s in verb_suffixes):
                    errors.append(
                        f'Sent {i}: V2 not VSO '
                        f'("{sent["v2"]}")'
                    )

            # V3: should have case, no VSO
            if '-nom' not in sent['v3']:
                errors.append(
                    f'Sent {i}: V3 missing -nom'
                )
            if '-acc' not in sent['v3']:
                # Only flag for transitive/ditransitive
                if sent['verb_type'] != 'intransitive':
                    errors.append(
                        f'Sent {i}: V3 missing -acc'
                    )

            # V4: should have case AND VSO
            if '-nom' not in sent['v4']:
                errors.append(
                    f'Sent {i}: V4 missing -nom'
                )
            if len(v4_tokens) > 1:
                if not any(v4_tokens[0].endswith(s)
                           for s in verb_suffixes):
                    errors.append(
                        f'Sent {i}: V4 not VSO'
                    )

            # V1 and V2 should NOT have case
            for v in ('v1', 'v2'):
                if '-nom' in sent[v] or '-acc' in sent[v]:
                    errors.append(
                        f'Sent {i}: {v} has case markers'
                    )

        if errors:
            for e in errors[:10]:
                self.errors.append(e)
            if len(errors) > 10:
                self.errors.append(
                    f'... and {len(errors) - 10} more'
                )
        else:
            print(
                f'  [PASS] All variant structures valid'
            )

    def check_word_counts(self):
        """Verify word counts are in expected range."""
        for i, sent in enumerate(self.data):
            wc = len(sent['v1'].split())
            if wc < 2 or wc > 8:
                self.warnings.append(
                    f'Sent {i}: unusual word count '
                    f'{wc} ("{sent["v1"]}")'
                )
            if wc != sent.get('word_count', wc):
                self.errors.append(
                    f'Sent {i}: stored word_count '
                    f'{sent["word_count"]} != actual {wc}'
                )
        print(
            f'  [PASS] Word counts in range (2-8)'
        )

    def check_fewshot_overlap(self):
        """Check that few-shot examples do not overlap
        with test sentences."""
        from language_generator import LanguageGenerator
        gen = LanguageGenerator(seed=42)
        overlaps = []
        for variant in ('v1', 'v2', 'v3', 'v4'):
            examples = gen.generate_few_shot_examples(
                variant, n=8
            )
            ex_set = {
                ex['translation'] for ex in examples
            }
            test_set = {
                s[variant] for s in self.data
            }
            overlap = ex_set & test_set
            if overlap:
                overlaps.append(
                    f'{variant}: {len(overlap)} shared '
                    f'sentences'
                )
        if overlaps:
            for o in overlaps:
                self.errors.append(
                    f'Few-shot/test overlap: {o}'
                )
        else:
            print(
                f'  [PASS] No few-shot/test overlap'
            )

    def print_samples(self, n: int = 5):
        """Show sample sentences for manual inspection."""
        print(f'\n--- Sample Sentences (first {n}) ---')
        for i, sent in enumerate(self.data[:n]):
            print(f'\n  [{i}] English: {sent["english"]}')
            print(f'      V1 (SVO, no case):  '
                  f'{sent["v1"]}')
            print(f'      V2 (VSO, no case):  '
                  f'{sent["v2"]}')
            print(f'      V3 (SVO, +case):    '
                  f'{sent["v3"]}')
            print(f'      V4 (VSO, +case):    '
                  f'{sent["v4"]}')
            print(f'      Type: {sent["verb_type"]}  '
                  f'| Tense: {sent["tense"]}')

    def print_summary_stats(self):
        """Print a quick statistical overview."""
        print(f'\n--- Dataset Summary ---')
        print(f'  Total sentences: {len(self.data)}')
        # Unique vocabulary
        all_words = set()
        for sent in self.data:
            all_words.update(sent['v1'].split())
        print(f'  Unique tokens (V1): {len(all_words)}')
        # Length stats
        lengths = [
            len(s['v1'].split()) for s in self.data
        ]
        print(f'  Sentence length: '
              f'min={min(lengths)}, '
              f'max={max(lengths)}, '
              f'mean={sum(lengths)/len(lengths):.1f}')
        # Adjective and pronoun coverage
        n_adj = sum(
            1 for s in self.data if s['has_adjective']
        )
        n_pro = sum(
            1 for s in self.data if s['has_pronoun']
        )
        print(f'  With adjectives: {n_adj}')
        print(f'  With pronouns: {n_pro}')

    def run_all(self, verbose: bool = False):
        """Run all verification checks."""
        print('=' * 50)
        print('DATASET VERIFICATION')
        print('=' * 50)
        if not self.load():
            self._print_results()
            return False

        print('\nRunning checks...')
        self.check_sentence_count()
        self.check_required_fields()
        self.check_verb_type_distribution()
        self.check_tense_distribution()
        self.check_duplicates()
        self.check_variant_structure()
        self.check_word_counts()
        self.check_fewshot_overlap()
        self.print_summary_stats()
        if verbose:
            self.print_samples(n=10)
        else:
            self.print_samples(n=3)
        self._print_results()
        return len(self.errors) == 0

    def _print_results(self):
        """Print final pass/fail summary."""
        print(f'\n{"=" * 50}')
        if self.errors:
            print(f'FAILED - {len(self.errors)} error(s):')
            for e in self.errors:
                print(f'  [ERROR] {e}')
        else:
            print('ALL CHECKS PASSED')
        if self.warnings:
            print(f'\n{len(self.warnings)} warning(s):')
            for w in self.warnings:
                print(f'  [WARN] {w}')
        print('=' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Verify generated dataset'
    )
    parser.add_argument(
        '--path',
        default='data/test_set.json',
        help='Path to test set JSON',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show more sample sentences',
    )
    args = parser.parse_args()
    verifier = DatasetVerifier(args.path)
    success = verifier.run_all(
        verbose=args.verbose
    )
    sys.exit(0 if success else 1)
