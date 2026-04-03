"""
Quick test script — runs 5 sentences across all 4 variants
and displays the prompt + model response side by side.

Usage:
    python quick_test.py --model smol
    python quick_test.py --model pythia --quantize 4bit
    python quick_test.py --model bloomz --quantize 8bit
"""

import argparse
import logging

from language_generator import LanguageGenerator
from prompt_builder import PromptBuilder
from model_inference import ModelInference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s',
)
logger = logging.getLogger('quick_test')

VARIANTS = ['v1', 'v2', 'v3', 'v4']
SEED = 42
N_SENTENCES = 5


def run_quick_test(model_key: str, quantize: str = '8bit'):
    gen = LanguageGenerator(seed=SEED)
    test_set = gen.generate_test_set()[:N_SENTENCES]

    logger.info(f'Loading {model_key} ({quantize})...')
    model = ModelInference(model_key, quantize=quantize)
    logger.info(f'VRAM usage: {model.get_vram_usage()}')

    for variant in VARIANTS:
        builder = PromptBuilder(variant)

        print(f'\n{"=" * 70}')
        print(f'  VARIANT: {variant}')
        print(f'{"=" * 70}')

        for i, sent in enumerate(test_set):
            gold = getattr(sent, variant)

            if model.model_type == 'instruction':
                prompt = builder.build_chat_prompt(sent.english)
            else:
                prompt = builder.build_prompt(sent.english)

            prediction = model.generate(prompt)

            print(f'\n--- Sentence {i + 1} ---')
            print(f'English:    {sent.english}')
            print(f'Expected:   {gold}')
            print(f'Prediction: {prediction}')
            match = prediction.strip().lower() == gold.strip().lower()
            print(f'Match:      {"YES" if match else "NO"}')

            print(f'\nPrompt sent to model:')
            print(f'  {prompt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quick test — 5 sentences x 4 variants'
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
    run_quick_test(args.model, args.quantize)
