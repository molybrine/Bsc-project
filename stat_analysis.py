"""
Module 5: Statistical Analysis
Factorial ANOVA, post-hoc tests, and learning curve regression.
"""
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyser:
    """Runs all statistical tests for the experiment."""

    ALPHA = 0.05

    def __init__(self, results_df: pd.DataFrame):
        """
        Args:
            results_df: DataFrame with columns:
                model, variant, shot_count, sentence_id,
                exact_match, edit_distance,
                word_order_correct, case_marking_correct
        """
        self.df = results_df
        self._add_factor_columns()

    def _add_factor_columns(self):
        """Derive factorial design columns."""
        self.df['syntax'] = self.df['variant'].map({
            'v1': 'SVO', 'v2': 'VSO',
            'v3': 'SVO', 'v4': 'VSO',
        })
        self.df['morphology'] = self.df['variant'].map({
            'v1': 'none', 'v2': 'none',
            'v3': 'case', 'v4': 'case',
        })
        # Unique subject IDs per model for between-
        # subjects ANOVA (avoids treating pythia_0
        # and bloomz_0 as the same subject)
        self.df['subject_id'] = (
            self.df['model'] + '_' +
            self.df['sentence_id'].astype(str)
        )

    def factorial_anova(self) -> Dict:
        """
        2x2 mixed ANOVA: syntax x morphology (within)
        x model (between).
        Tests H2 (feature difficulty and interaction).
        Falls back to rm_anova if only one model is present.
        """
        models = self.df['model'].unique()
        if len(models) >= 2:
            aov = pg.mixed_anova(
                data=self.df,
                dv='exact_match',
                within=['syntax', 'morphology'],
                between='model',
                subject='subject_id',
            )
        else:
            logger.info(
                'Only one model — running repeated-measures '
                'ANOVA (no between factor)'
            )
            aov = pg.rm_anova(
                data=self.df,
                dv='exact_match',
                within=['syntax', 'morphology'],
                subject='subject_id',
            )
        logger.info('Factorial ANOVA results:')
        logger.info(f'\n{aov.to_string()}')

        def _is_sig(source_name):
            row = aov.loc[aov['Source'] == source_name]
            if row.empty:
                return False
            return row['p-unc'].values[0] < self.ALPHA

        return {
            'anova_table': aov,
            'syntax_sig': _is_sig('syntax'),
            'morphology_sig': _is_sig('morphology'),
            'interaction_sig': _is_sig(
                'syntax * morphology'
            ),
        }

    def post_hoc_tests(self) -> pd.DataFrame:
        """Pairwise comparisons with Bonferroni correction."""
        pairs = pg.pairwise_tests(
            data=self.df,
            dv='exact_match',
            within='variant',
            subject='subject_id',
            padjust='bonf',
        )
        return pairs

    @staticmethod
    def _log_curve(x, a, b, c):
        """Logarithmic model: y = a * log(x + 1) + c"""
        return a * np.log(x + 1) + c

    def fit_learning_curves(self) -> Dict:
        """
        Fit logarithmic regression to learning curves.
        Tests H3 (diminishing returns).
        """
        results = {}
        for model in self.df['model'].unique():
            for variant in self.df['variant'].unique():
                subset = self.df[
                    (self.df['model'] == model) &
                    (self.df['variant'] == variant)
                ]
                x = subset.groupby('shot_count')[
                    'exact_match'].mean()
                shots = np.array(x.index)
                accs = np.array(x.values)
                try:
                    popt, pcov = curve_fit(
                        self._log_curve,
                        shots, accs,
                        p0=[0.2, 1.0, 0.3],
                        maxfev=5000,
                    )
                    perr = np.sqrt(np.diag(pcov))
                    results[f'{model}_{variant}'] = {
                        'params': popt,
                        'std_err': perr,
                        'r_squared': 1 - (
                            np.sum((accs - self._log_curve(
                                shots, *popt)) ** 2) /
                            np.sum((accs - accs.mean()) ** 2)
                        ),
                    }
                except RuntimeError:
                    logger.warning(
                        f'Curve fit failed: {model}/{variant}'
                    )
        return results

    def compute_effect_sizes(self) -> Dict:
        """Cohen's d for key pairwise comparisons."""
        effects = {}
        # Morphology vs no morphology
        case = self.df[self.df['morphology'] == 'case']
        no_case = self.df[self.df['morphology'] == 'none']
        effects['morphology_d'] = pg.compute_effsize(
            case['exact_match'],
            no_case['exact_match'],
            eftype='cohen',
        )
        # Between-model comparison (only if 2+ models)
        models = self.df['model'].unique()
        if len(models) >= 2:
            m1 = self.df[self.df['model'] == models[0]]
            m2 = self.df[self.df['model'] == models[1]]
            effects['model_d'] = pg.compute_effsize(
                m1['exact_match'],
                m2['exact_match'],
                eftype='cohen',
            )
        else:
            logger.info(
                'Only one model — skipping between-model '
                'effect size'
            )
            effects['model_d'] = None
        return effects
