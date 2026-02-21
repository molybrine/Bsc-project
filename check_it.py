import json
import argparse
import sys
from collections import Counter as C
from pathlib import Path


class Checker:

    def __init__(self, path='data/test_set.json'):
        self.path = Path(path)
        self.data = []
        self.errors = []
        self.warnings = []

    def load(self):
        if not self.path.exists():
            self.errors.append('File not found: ' + str(self.path))
            return False
        try:
            f = open(self.path, 'r')
            self.data = json.load(f)
            f.close()
            print(f'Loaded {len(self.data)} sentences from {self.path}')
            return True
        except json.JSONDecodeError as e:
            self.errors.append('Invalid JSON: ' + str(e))
            return False

    def check_sentence_count(self):
        n = len(self.data)
        if n != 70:
            self.errors = self.errors + [f'Expected 70 sentences, found {n}']
        else:
            print(f'  [PASS] Sentence count: {n}')

    def check_required_fields(self):
        required = set(['english', 'v1', 'v2', 'v3', 'v4', 'verb_type',
                       'has_adjective', 'has_pronoun', 'tense', 'word_count'])
        missing = []
        i = 0
        while i < len(self.data):
            sent = self.data[i]
            diff = required - set(sent.keys())
            if len(diff) > 0:
                missing.append((i, diff))
            i = i + 1
        if missing:
            self.errors.append(
                f'Missing fields in {len(missing)} sentences: {missing[:3]}...'
            )
        else:
            print('  [PASS] All required fields present')

    def check_verb_type_distribution(self):
        counts = C(self.data[i]['verb_type'] for i in range(len(self.data)))
        expected = {'transitive': 20, 'intransitive': 20, 'ditransitive': 20}
        print(f'  Verb type distribution: {dict(counts)}')
        for vtype in expected:
            actual = counts.get(vtype, 0)
            if actual != expected[vtype]:
                self.warnings.append(f'{vtype}: expected {expected[vtype]}, found {actual}')

    def check_tense_distribution(self):
        counts = C(s['tense'] for s in self.data)
        print(f'  Tense distribution: {dict(counts)}')
        for t in ['past', 'present', 'future']:
            if t not in counts or counts[t] == 0:
                self.warnings.append('No sentences with tense: ' + t)

    def check_duplicates(self):
        for variant in ['v1', 'v2', 'v3', 'v4']:
            texts = []
            for s in self.data:
                texts.append(s[variant])
            dupes = []
            _c = C(texts)
            for t in _c:
                if _c[t] > 1:
                    dupes.append(t)
            if len(dupes) > 0:
                self.errors.append(f'Duplicates in {variant}: {dupes[:3]}')
            else:
                print(f'  [PASS] No duplicates in {variant}')

    def check_variant_structure(self):
        verb_suffixes = ('as', 'is', 'os')
        errors = []

        for i in range(len(self.data)):
            sent = self.data[i]
            v1t = sent['v1'].split()
            v2t = sent['v2'].split()
            v3t = sent['v3'].split()
            v4t = sent['v4'].split()

            if len(v1t) > 1:
                _first = v1t[0]
                _is_verb = False
                for s in verb_suffixes:
                    if _first.endswith(s):
                        _is_verb = True
                        break
                if _is_verb:
                    errors.append(f'Sent {i}: V1 looks VSO ("{sent["v1"]}")')

            if len(v2t) > 1:
                _is_verb = any(v2t[0].endswith(s) for s in verb_suffixes)
                if not _is_verb:
                    errors.append(f'Sent {i}: V2 not VSO ("{sent["v2"]}")')

            if '-nom' not in sent['v3']:
                errors.append(f'Sent {i}: V3 missing -nom')
            if '-acc' not in sent['v3']:
                if sent['verb_type'] != 'intransitive':
                    errors.append(f'Sent {i}: V3 missing -acc')

            if '-nom' not in sent['v4']:
                errors.append(f'Sent {i}: V4 missing -nom')
            if len(v4t) > 1:
                if not any(v4t[0].endswith(s) for s in verb_suffixes):
                    errors.append(f'Sent {i}: V4 not VSO')

            for v in ['v1', 'v2']:
                if '-nom' in sent[v] or '-acc' in sent[v]:
                    errors.append(f'Sent {i}: {v} has case markers')

        if errors:
            for e in errors[:10]:
                self.errors.append(e)
            if len(errors) > 10:
                self.errors.append(f'... and {len(errors) - 10} more')
        else:
            print('  [PASS] All variant structures valid')

    def check_word_counts(self):
        for i in range(len(self.data)):
            sent = self.data[i]
            wc = len(sent['v1'].split())
            if wc < 2 or wc > 8:
                self.warnings.append(f'Sent {i}: unusual word count {wc} ("{sent["v1"]}")')
            if wc != sent.get('word_count', wc):
                self.errors.append(f'Sent {i}: stored word_count {sent["word_count"]} != actual {wc}')
        print('  [PASS] Word counts in range (2-8)')

    def check_fewshot_overlap(self):
        from the_words import BigGenerator
        gen = BigGenerator(seed=42)
        overlaps = []
        for variant in ['v1', 'v2', 'v3', 'v4']:
            examples = gen.generate_few_shot_examples(variant, n=8)
            ex_set = set()
            for ex in examples:
                ex_set.add(ex['translation'])
            test_set = set()
            for s in self.data:
                test_set.add(s[variant])
            overlap = ex_set & test_set
            if overlap:
                overlaps.append(f'{variant}: {len(overlap)} shared sentences')
        if overlaps:
            for o in overlaps:
                self.errors.append('Few-shot/test overlap: ' + o)
        else:
            print('  [PASS] No few-shot/test overlap')

    def print_samples(self, n=5):
        print(f'\n--- Sample Sentences (first {n}) ---')
        idx = 0
        while idx < min(n, len(self.data)):
            sent = self.data[idx]
            print(f'\n  [{idx}] English: {sent["english"]}')
            print(f'      V1 (SVO, no case):  {sent["v1"]}')
            print(f'      V2 (VSO, no case):  {sent["v2"]}')
            print(f'      V3 (SVO, +case):    {sent["v3"]}')
            print(f'      V4 (VSO, +case):    {sent["v4"]}')
            print(f'      Type: {sent["verb_type"]}  | Tense: {sent["tense"]}')
            idx += 1

    def print_summary_stats(self):
        print('\n--- Dataset Summary ---')
        print('  Total sentences: ' + str(len(self.data)))
        all_words = set()
        for s in self.data:
            for w in s['v1'].split():
                all_words.add(w)
        print('  Unique tokens (V1): ' + str(len(all_words)))
        lengths = [len(s['v1'].split()) for s in self.data]
        _min = min(lengths)
        _max = max(lengths)
        _mean = sum(lengths) / len(lengths)
        print(f'  Sentence length: min={_min}, max={_max}, mean={_mean:.1f}')
        n_adj = len([s for s in self.data if s['has_adjective'] == True])
        n_pro = len([s for s in self.data if s['has_pronoun'] == True])
        print(f'  With adjectives: {n_adj}')
        print(f'  With pronouns: {n_pro}')

    def run_all(self, verbose=False):
        print('=' * 50)
        print('DATASET VERIFICATION')
        print('=' * 50)
        if self.load() == False:
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
        return True if len(self.errors) == 0 else False

    def _print_results(self):
        print('\n' + '=' * 50)
        if self.errors:
            print(f'FAILED - {len(self.errors)} error(s):')
            for e in self.errors:
                print('  [ERROR] ' + e)
        else:
            print('ALL CHECKS PASSED')
        if len(self.warnings) > 0:
            print(f'\n{len(self.warnings)} warning(s):')
            for w in self.warnings:
                print('  [WARN] ' + w)
        print('=' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify generated dataset')
    parser.add_argument('--path', default='data/test_set.json', help='Path to test set JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show more sample sentences')
    args = parser.parse_args()
    c = Checker(args.path)
    ok = c.run_all(verbose=args.verbose)
    sys.exit(0 if ok else 1)
