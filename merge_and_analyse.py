"""
Merge per-model results and run full analysis.

Usage:
    python merge_and_analyse.py \
        --pythia results/results_pythia_20250601.csv \
        --bloomz results/results_bloomz_20250601.csv
"""

import argparse
import pandas as pd
import logging
from pathlib import Path

from stat_analysis import StatisticalAnalyser
from visualisation import Visualiser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s',
)
logger = logging.getLogger('analysis')


def merge_and_analyse(
    pythia_path: str, bloomz_path: str
):
    """Combine results and run full analysis."""

    # Step 1: Load and merge
    logger.info('Loading result files...')
    df_pythia = pd.read_csv(pythia_path)
    df_bloomz = pd.read_csv(bloomz_path)
    df = pd.concat(
        [df_pythia, df_bloomz], ignore_index=True
    )
    logger.info(
        f'Merged: {len(df_pythia)} + {len(df_bloomz)}'
        f' = {len(df)} total rows'
    )

    # Sanity check
    expected = 2 * 4 * 3 * 70  # 1680
    if len(df) != expected:
        logger.warning(
            f'Expected {expected} rows, got {len(df)}'
        )

    # Save merged file
    merged_path = Path('results/results_merged.csv')
    df.to_csv(merged_path, index=False)
    logger.info(f'Merged results saved to {merged_path}')

    # Step 2: Statistical analysis
    logger.info('Running statistical analysis...')
    analyser = StatisticalAnalyser(df)

    anova = analyser.factorial_anova()
    logger.info(
        f"Syntax significant: {anova['syntax_sig']}"
    )
    logger.info(
        f"Morphology significant: "
        f"{anova['morphology_sig']}"
    )
    logger.info(
        f"Interaction significant: "
        f"{anova['interaction_sig']}"
    )

    effects = analyser.compute_effect_sizes()
    logger.info(f'Effect sizes: {effects}')

    post_hoc = analyser.post_hoc_tests()
    logger.info(f'Post-hoc results:\n{post_hoc}')

    curves = analyser.fit_learning_curves()
    for key, val in curves.items():
        logger.info(
            f"  {key}: R2={val['r_squared']:.3f}"
        )

    # Step 3: Generate figures
    logger.info('Generating figures...')
    viz = Visualiser(df, output_dir='figures/')
    viz.generate_all()

    # Final summary
    logger.info('=' * 50)
    logger.info('ANALYSIS COMPLETE')
    for model in ['pythia', 'bloomz']:
        m_acc = df[
            df['model'] == model
        ]['exact_match'].mean()
        logger.info(f'  {model}: {m_acc:.3f}')
    logger.info('=' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge results and analyse'
    )
    parser.add_argument(
        '--pythia', required=True,
        help='Path to Pythia results CSV',
    )
    parser.add_argument(
        '--bloomz', required=True,
        help='Path to BLOOMZ results CSV',
    )
    args = parser.parse_args()
    merge_and_analyse(args.pythia, args.bloomz)
