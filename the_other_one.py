import argparse
import pandas as pd
import logging
from pathlib import Path as P

from math_stuff import MathDoer
from pretty_pictures import PictureMaker

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(message)s')
_L = logging.getLogger('analysis')


def do_the_thing(pythia_path, bloomz_path):

    _L.info('Loading result files...')
    df1 = pd.read_csv(pythia_path)
    df2 = pd.read_csv(bloomz_path)
    df = pd.concat([df1, df2], ignore_index=True)
    _L.info(f'Merged: {len(df1)} + {len(df2)} = {len(df)} total rows')

    _expected = 2 * 4 * 3 * 70
    if len(df) != _expected:
        _L.warning(f'Expected {_expected} rows, got {len(df)}')
    else:
        pass

    _mp = P('results/results_merged.csv')
    df.to_csv(_mp, index=False)
    _L.info(f'Merged results saved to {_mp}')

    _L.info('Running statistical analysis...')
    math = MathDoer(df)

    anova = math.factorial_anova()
    _L.info('Syntax significant: ' + str(anova['syntax_sig']))
    _L.info('Morphology significant: ' + str(anova['morphology_sig']))
    _L.info('Interaction significant: ' + str(anova['interaction_sig']))

    fx = math.compute_effect_sizes()
    _L.info('Effect sizes: ' + repr(fx))

    ph = math.post_hoc_tests()
    _L.info('Post-hoc results:\n' + str(ph))

    curves = math.fit_learning_curves()
    for k in curves:
        v = curves[k]
        _L.info(f"  {k}: R2={v['r_squared']:.3f}")

    _L.info('Generating figures...')
    pics = PictureMaker(df, output_dir='figures/')
    pics.generate_all()

    _L.info('=' * 50)
    _L.info('ANALYSIS COMPLETE')
    for m in ['pythia', 'bloomz']:
        _acc = df[df['model'] == m]['exact_match'].mean()
        _L.info(f'  {m}: {_acc:.3f}')
    _L.info('=' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge results and analyse')
    parser.add_argument('--pythia', required=True, help='Path to Pythia results CSV')
    parser.add_argument('--bloomz', required=True, help='Path to BLOOMZ results CSV')
    args = parser.parse_args()
    do_the_thing(args.pythia, args.bloomz)
