"""
Merge per-model results and run full analysis.

Usage:
    python merge_and_analyse.py \
        --model1 results/results_pythia_20250601.csv \
        --model2 results/results_bloomz_20250601.csv
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
    path1: str, path2: str
):
    """Combine results and run full analysis."""

    # Step 1: Load and merge
    logger.info('Loading result files...')
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df = pd.concat(
        [df1, df2], ignore_index=True
    )
    logger.info(
        f'Merged: {len(df1)} + {len(df2)}'
        f' = {len(df)} total rows'
    )

    # Sanity check
    n_models = df['model'].nunique()
    n_variants = df['variant'].nunique()
    n_shots = df['shot_count'].nunique()
    logger.info(
        f'Found {n_models} model(s), '
        f'{n_variants} variant(s), '
        f'{n_shots} shot level(s)'
    )

    # Save merged file
    merged_path = Path('results/results_merged.csv')
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(merged_path, index=False)
    logger.info(f'Merged results saved to {merged_path}')

    # Step 2: Statistical analysis
    logger.info('Running statistical analysis...')
    try:
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
    except Exception as e:
        logger.warning(
            f'Statistical analysis failed: {e}'
        )
        logger.warning(
            'Continuing to figure generation...'
        )

    # Step 3: Generate figures
    logger.info('Generating figures...')
    try:
        viz = Visualiser(df, output_dir='figures/')
        viz.generate_all()
    except Exception as e:
        logger.error(f'Figure generation failed: {e}')

    # Final summary
    logger.info('=' * 50)
    logger.info('ANALYSIS COMPLETE')
    for model in df['model'].unique():
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
        '--model1', required=True,
        help='Path to first model results CSV',
    )
    parser.add_argument(
        '--model2', required=True,
        help='Path to second model results CSV',
    )
    args = parser.parse_args()
    merge_and_analyse(args.model1, args.model2)
