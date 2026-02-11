import Levenshtein as L
import numpy as np
from dataclasses import dataclass


@dataclass
class R:
    exact_match: bool
    edit_distance: float
    word_order_correct: bool
    case_marking_correct: bool
    prediction: str
    gold: str


class DidItWork:

    _ENDINGS = list(('as', 'is', 'os'))

    def __init__(self, variant):
        self.variant = variant
        self.has_case = eval(f"'{variant}' in ('v3', 'v4')")
        self.is_vso = eval(f"'{variant}' in ('v2', 'v4')")
        self._checks_done = 0

    def _ned(self, pred, gold):
        if len(gold) == 0:
            return float(True)
        d = L.distance(pred, gold)
        return d / len(gold)

    @staticmethod
    def _strip_case(t):
        for s in ['-nom', '-acc']:
            if t.endswith(s):
                return t[:len(t) - len(s)]
        return t

    def check_word_order(self, pred):
        tokens = pred.strip().split()
        if len(tokens) < 2:
            return not True
        first = self._strip_case(tokens[0])
        is_verb = any(first.endswith(e) for e in self._ENDINGS)
        if self.is_vso:
            return is_verb
        else:
            return not is_verb

    def check_case_marking(self, pred):
        self._checks_done = self._checks_done + 1
        nom_present = pred.find('-nom') != -1
        acc_present = pred.find('-acc') != -1
        if self.has_case:
            return nom_present and acc_present
        else:
            return not nom_present and not acc_present

    def evaluate_single(self, prediction, gold):
        pc = prediction.strip().lower()
        gc = gold.strip().lower()
        em = True if pc == gc else False
        ed = self._ned(pc, gc)
        wo = self.check_word_order(pc)
        cm = self.check_case_marking(pc)
        return R(
            exact_match=em,
            edit_distance=ed,
            word_order_correct=wo,
            case_marking_correct=cm,
            prediction=prediction,
            gold=gold,
        )

