"""
Quick test script — runs N sentences across all 4 variants
and displays the model's output for each, with a per-variant
accuracy summary at the end.

Usage:
    python quick_test.py --model smol
    python quick_test.py --model pythia --quantize 4bit
    python quick_test.py --model bloomz --shots 3
    python quick_test.py --model smol --n 10 --verbose
"""

import argparse
import logging
import sys

from language_generator import LanguageGenerator
from prompt_builder import PromptBuilder
from model_inference import ModelInference
from evaluation import Evaluator

# Try to use UTF-8 for output (needed on Windows cp1252 terminals).
# If that fails, we'll fall back to ASCII characters below.
_UTF8 = True
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Python 3.7+
except (AttributeError, Exception):
    _UTF8 = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s',
)
logger = logging.getLogger('quick_test')

VARIANTS = ['v1', 'v2', 'v3', 'v4']
VARIANT_LABELS = {
    'v1': 'SVO, no case marking',
    'v2': 'VSO, no case marking',
    'v3': 'SVO, case marking',
    'v4': 'VSO, case marking',
}
SEED = 42

# Characters used for output — use ASCII fallbacks if UTF-8 is unavailable
TICK = '✓' if _UTF8 else '[OK]'
CROSS = '✗' if _UTF8 else '[X] '
BAR_FULL = '█' if _UTF8 else '#'
BAR_EMPTY = '░' if _UTF8 else '.'
BOX_V = '│' if _UTF8 else '|'
BOX_H = '─' if _UTF8 else '-'


class Colour:
    """ANSI colour codes. Works on most modern terminals."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    DIM = '\033[90m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        """Replace all codes with empty strings."""
        for attr in ('GREEN', 'RED', 'YELLOW', 'CYAN', 'BOLD', 'DIM', 'RESET'):
            setattr(cls, attr, '')


def _accuracy_colour(pct: float) -> str:
    """Colour-code an accuracy percentage."""
    if pct < 25:
        return Colour.RED
    elif pct < 70:
        return Colour.YELLOW
    return Colour.GREEN


def _progress_bar(fraction: float, width: int = 20) -> str:
    """Build a progress bar coloured by accuracy."""
    filled = int(round(fraction * width))
    empty = width - filled
    colour = _accuracy_colour(fraction * 100)
    return f'{colour}{BAR_FULL * filled}{Colour.DIM}{BAR_EMPTY * empty}{Colour.RESET}'


def _print_variant_header(variant: str):
    label = VARIANT_LABELS[variant]
    print()
    print(f'{Colour.CYAN}{Colour.BOLD}{"=" * 70}')
    print(f'  VARIANT {variant} — {label}')
    print(f'{"=" * 70}{Colour.RESET}')


def _print_prompt_template(prompt: str, english: str):
    """Show the prompt once per variant (with <SENTENCE> placeholder)."""
    # Replace the actual sentence with a placeholder to show the template
    template = prompt.replace(english, '<SENTENCE>')
    print(f'\n  {Colour.DIM}Prompt template:{Colour.RESET}')
    for line in template.splitlines():
        print(f'  {Colour.DIM}{BOX_V} {line}{Colour.RESET}')


def _print_sentence(
    index: int, english: str, gold: str,
    prediction: str, is_match: bool, model_label: str,
):
    if is_match:
        tick = f'{Colour.GREEN}{TICK}{Colour.RESET}'
    else:
        tick = f'{Colour.RED}{CROSS}{Colour.RESET}'
    # Show empty predictions explicitly
    pred_display = prediction if prediction.strip() else f'{Colour.DIM}(empty){Colour.RESET}'
    print(f'\n  [{index}] English:      {english}')
    print(f'      Expected:     {gold}')
    print(f'      {model_label}: {pred_display}  {tick}')


def _print_summary(
    results: dict, model_key: str, n_sentences: int,
):
    print()
    print(f'{Colour.CYAN}{Colour.BOLD}{"=" * 70}')
    print(f'  SUMMARY — {model_key}')
    print(f'{"=" * 70}{Colour.RESET}')
    print()

    total_correct = 0
    total_seen = 0
    for variant in VARIANTS:
        correct = results[variant]
        pct = (correct / n_sentences) * 100
        label = f'{variant} ({VARIANT_LABELS[variant]})'
        bar = _progress_bar(correct / n_sentences)
        colour = _accuracy_colour(pct)
        print(
            f'  {label:<32} '
            f'{correct}/{n_sentences}  '
            f'{colour}({pct:5.1f}%){Colour.RESET}  '
            f'{bar}'
        )
        total_correct += correct
        total_seen += n_sentences

    print(f'  {Colour.DIM}{BOX_H * 60}{Colour.RESET}')
    total_pct = (total_correct / total_seen) * 100
    colour = _accuracy_colour(total_pct)
    print(
        f'  {"TOTAL":<32} '
        f'{total_correct}/{total_seen} '
        f'{colour}({total_pct:5.1f}%){Colour.RESET}'
    )
    print()


def run_quick_test(
    model_key: str,
    quantize: str = '8bit',
    n: int = 5,
    shots: int = 0,
    verbose: bool = False,
):
    """Run N sentences through all 4 variants and print results."""
    gen = LanguageGenerator(seed=SEED)
    test_set = gen.generate_test_set()[:n]

    logger.info(f'Loading {model_key} ({quantize})...')
    model = ModelInference(model_key, quantize=quantize)
    logger.info(f'VRAM usage: {model.get_vram_usage()}')
    logger.info(
        f'Running {n} sentences x 4 variants ({shots}-shot)'
    )

    model_label = f'{model_key.capitalize()} output'
    results = {v: 0 for v in VARIANTS}

    for variant in VARIANTS:
        builder = PromptBuilder(variant)
        evaluator = Evaluator(variant)
        examples = None
        if shots > 0:
            examples = gen.generate_few_shot_examples(variant, n=shots)

        _print_variant_header(variant)

        first_prompt_shown = False
        for i, sent in enumerate(test_set):
            gold = getattr(sent, variant)
            prompt = builder.build_prompt(sent.english, examples=examples)

            if verbose and not first_prompt_shown:
                _print_prompt_template(prompt, sent.english)
                first_prompt_shown = True

            prediction = model.generate(prompt)
            result = evaluator.evaluate_single(prediction, gold)

            if result.exact_match:
                results[variant] += 1

            _print_sentence(
                index=i + 1,
                english=sent.english,
                gold=gold,
                prediction=prediction,
                is_match=result.exact_match,
                model_label=model_label,
            )

    _print_summary(results, model_key, n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quick test — N sentences across all 4 variants',
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
    parser.add_argument(
        '--n', type=int, default=5,
        help='Number of test sentences per variant (default: 5)',
    )
    parser.add_argument(
        '--shots', type=int, default=0,
        choices=[0, 3, 8],
        help='Few-shot examples to include (default: 0)',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show the prompt template for each variant',
    )
    parser.add_argument(
        '--no-color', action='store_true',
        help='Disable colour output',
    )
    args = parser.parse_args()

    # Disable colours if requested or if not writing to a terminal
    if args.no_color or not sys.stdout.isatty():
        Colour.disable()

    run_quick_test(
        model_key=args.model,
        quantize=args.quantize,
        n=args.n,
        shots=args.shots,
        verbose=args.verbose,
    )
