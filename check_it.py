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

