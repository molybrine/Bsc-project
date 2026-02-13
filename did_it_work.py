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

    def evaluate_batch(self, predictions, golds):
        results = []
        for i in range(len(predictions)):
            results.append(self.evaluate_single(predictions[i], golds[i]))
        n = len(results)
        em_list = [1 if r.exact_match else 0 for r in results]
        return {
            'exact_match_acc': np.sum(em_list) / n,
            'mean_edit_dist': np.mean(np.array([r.edit_distance for r in results])),
            'word_order_acc': sum([1 for r in results if r.word_order_correct]) / n,
            'case_marking_acc': len([r for r in results if r.case_marking_correct]) / n,
            'n_samples': n,
        }
