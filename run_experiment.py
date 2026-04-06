"""
Module 7: Main Experiment Runner
Runs all conditions for a single model.

Usage:
    python run_experiment.py --model pythia
    python run_experiment.py --model bloomz
    python run_experiment.py --model smol
    python run_experiment.py --model pythia --quantize 4bit
    python run_experiment.py --model pythia --quantize none
"""

import argparse
import json
import logging
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from language_generator import LanguageGenerator
from prompt_builder import PromptBuilder
from model_inference import ModelInference
from evaluation import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s',
)
logger = logging.getLogger('experiment')

# -- Configuration --
VARIANTS = ['v1', 'v2', 'v3', 'v4']
SHOT_CONDITIONS = [0, 3, 8]
SEED = 42
# Use environment variable if set, otherwise default
OUTPUT_DIR = Path(
    os.environ.get('RESULTS_DIR', 'results/')
)


def run_experiment(model_key: str, quantize: str = '8bit'):
    """Run all conditions for a single model."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Step 1: Load or generate data
    data_path = Path('data/test_set.json')
    gen = LanguageGenerator(seed=SEED)
    if data_path.exists():
        logger.info(
            f'Loading existing test set from {data_path}'
        )
        with open(data_path) as f:
            raw = json.load(f)
        test_set = gen.generate_test_set()
    else:
        logger.info('Generating test data...')
        test_set = gen.save_dataset()

    # Step 2: Load model
    logger.info(f'Loading {model_key} ({quantize})...')
    model = ModelInference(
        model_key, quantize=quantize
    )
    logger.info(
        f'VRAM usage: {model.get_vram_usage()}'
    )

    # Step 3: Run all variant x shot conditions
    all_results = []
    total = len(VARIANTS) * len(SHOT_CONDITIONS) * 70
    logger.info(
        f'Running {len(VARIANTS)} variants x '
        f'{len(SHOT_CONDITIONS)} shots x 70 sentences '
        f'= {total} instances'
    )

    for variant in VARIANTS:
        builder = PromptBuilder(variant)
        evaluator = Evaluator(variant)
        examples_pool = gen.generate_few_shot_examples(
            variant, n=8
        )

        for n_shots in SHOT_CONDITIONS:
            examples = examples_pool[:n_shots]
            logger.info(
                f'Running: {model_key} | {variant}'
                f' | {n_shots}-shot'
            )

            for i, sent in enumerate(
                tqdm(test_set,
                     desc=f'{variant}/{n_shots}-shot')
            ):
                prompt = builder.build_prompt(
                    sent.english,
                    examples if n_shots > 0
                    else None,
                )
                prediction = model.generate(prompt)
                gold = getattr(sent, variant)
                result = evaluator.evaluate_single(
                    prediction, gold
                )

                all_results.append({
                    'model': model_key,
                    'quantize': quantize,
                    'variant': variant,
                    'shot_count': n_shots,
                    'sentence_id': i,
                    'english': sent.english,
                    'gold': gold,
                    'prediction': prediction,
                    'exact_match': result.exact_match,
                    'edit_distance':
                        result.edit_distance,
                    'word_order_correct':
                        result.word_order_correct,
                    'case_marking_correct':
                        result.case_marking_correct,
                })

    # Step 4: Save results for this model
    df = pd.DataFrame(all_results)
    csv_path = (
        OUTPUT_DIR /
        f'results_{model_key}_{quantize}_{timestamp}.csv'
    )
    df.to_csv(csv_path, index=False)
    logger.info(f'Results saved to {csv_path}')

    # Quick summary
    logger.info('=' * 50)
    logger.info(f'{model_key.upper()} COMPLETE')
    logger.info(f'Instances: {len(df)}')
    logger.info(
        f"Accuracy: "
        f"{df['exact_match'].mean():.3f}"
    )
    logger.info('=' * 50)
    return csv_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run experiment for one model'
    )
    parser.add_argument(
        '--model', required=True,
        choices=['pythia', 'bloomz', 'smol'],
        help='Which model to run',
    )
    parser.add_argument(
        '--quantize', default='8bit',
        choices=['8bit', '4bit', 'none'],
        help='Quantisation level (default: 8bit)',
    )
    args = parser.parse_args()
    run_experiment(args.model, args.quantize)
