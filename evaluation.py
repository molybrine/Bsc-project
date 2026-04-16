"""
Module 4: Evaluation Metrics
Implements exact match, edit distance, and feature-level accuracy.
"""
import Levenshtein
import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Evaluation result for a single prediction."""
    exact_match: bool
    edit_distance: float        # normalised 0-1
    word_order_correct: bool
    case_marking_correct: bool
    prediction: str
    gold: str


class Evaluator:
    """Computes all three evaluation metrics."""

    def __init__(self, variant: str):
        self.variant = variant
        self.has_case = variant in ('v3', 'v4')
        self.is_vso = variant in ('v2', 'v4')

    def normalised_edit_distance(
        self, pred: str, gold: str
    ) -> float:
        """Character-level Levenshtein / len(gold)."""
        if not gold:
            return 1.0
        dist = Levenshtein.distance(pred, gold)
        return dist / len(gold)

    @staticmethod
    def _strip_case(token: str) -> str:
        """Remove case suffixes from a token."""
        for suffix in ('-nom', '-acc'):
            if token.endswith(suffix):
                return token[:-len(suffix)]
        return token

    def check_word_order(
        self, pred: str
    ) -> bool:
        """Check if predicted word order matches variant."""
        tokens = pred.strip().split()
        if len(tokens) < 2:
            return False
        first = self._strip_case(tokens[0])
        if self.is_vso:
            # First token should be the verb
            return any(
                first.endswith(suf)
                for suf in ('as', 'is', 'os')
            )
        else:
            # First token should NOT be the verb
            return not any(
                first.endswith(suf)
                for suf in ('as', 'is', 'os')
            )

    def check_case_marking(
        self, pred: str
    ) -> bool:
        """Check case suffixes if variant requires them."""
        if not self.has_case:
            # Should NOT have case markers
            return '-nom' not in pred and '-acc' not in pred
        # Should have both markers
        return '-nom' in pred and '-acc' in pred

    def evaluate_single(
        self, prediction: str, gold: str
    ) -> EvalResult:
        """Evaluate one prediction against gold standard."""
        pred_clean = prediction.strip().lower()
        gold_clean = gold.strip().lower()
        return EvalResult(
            exact_match=(pred_clean == gold_clean),
            edit_distance=self.normalised_edit_distance(
                pred_clean, gold_clean
            ),
            word_order_correct=self.check_word_order(
                pred_clean
            ),
            case_marking_correct=self.check_case_marking(
                pred_clean
            ),
            prediction=prediction,
            gold=gold,
        )

    def evaluate_batch(
        self,
        predictions: List[str],
        golds: List[str],
    ) -> Dict[str, float]:
        """Compute aggregate metrics over a batch."""
        results = [
            self.evaluate_single(p, g)
            for p, g in zip(predictions, golds)
        ]
        n = len(results)
        return {
            'exact_match_acc': sum(
                r.exact_match for r in results) / n,
            'mean_edit_dist': np.mean(
                [r.edit_distance for r in results]),
            'word_order_acc': sum(
                r.word_order_correct for r in results) / n,
            'case_marking_acc': sum(
                r.case_marking_correct for r in results) / n,
            'n_samples': n,
        }
