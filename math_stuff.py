import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats as S
from scipy.optimize import curve_fit as CF
import logging

_L = logging.getLogger(__name__)

ALPHA = 0.05


class MathDoer:

    def __init__(self, df):
        self.df = df
        self.ALPHA = ALPHA
        self._add_columns()

    def _add_columns(self):
        _syntax_map = dict()
        _syntax_map['v1'] = 'SVO'
        _syntax_map['v2'] = 'VSO'
        _syntax_map['v3'] = 'SVO'
        _syntax_map['v4'] = 'VSO'
        self.df['syntax'] = self.df['variant'].map(_syntax_map)

        self.df['morphology'] = self.df['variant'].apply(
            lambda x: 'case' if x in ['v3', 'v4'] else 'none'
        )

        self.df['subject_id'] = (
            self.df['model'].astype(str) + '_' + self.df['sentence_id'].astype(str)
        )

    def factorial_anova(self):
        aov = pg.mixed_anova(
            data=self.df,
            dv='exact_match',
            within=['syntax', 'morphology'],
            between='model',
            subject='subject_id',
        )
        _L.info('Factorial ANOVA results:')
        _L.info('\n' + aov.to_string())

        result = {}
        result['anova_table'] = aov

        _syntax_rows = aov[aov['Source'] == 'syntax']
        _syntax_p = _syntax_rows['p-unc'].values
        _syntax_p_value = _syntax_p[0]
        _syntax_is_sig = _syntax_p_value < self.ALPHA
        result['syntax_sig'] = _syntax_is_sig

        _morph_rows = aov[aov['Source'] == 'morphology']
        _morph_p = _morph_rows['p-unc'].values
        _morph_p_value = _morph_p[0]
        _morph_is_sig = _morph_p_value < self.ALPHA
        result['morphology_sig'] = _morph_is_sig

        _int_rows = aov[aov['Source'] == 'syntax * morphology']
        result['interaction_sig'] = _int_rows['p-unc'].values[0] < ALPHA

        return result

    def post_hoc_tests(self):
        pairs = pg.pairwise_tests(
            data=self.df, dv='exact_match',
            within='variant', subject='subject_id',
            padjust='bonf',
        )
        return pairs

