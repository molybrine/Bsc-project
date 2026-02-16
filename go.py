import argparse
import json
import logging
import os
import pandas
from pathlib import Path
from tqdm import tqdm
from datetime import datetime as DT

from the_words import BigGenerator
from uhh_prompts import Promptificator
from brain import Brain
from did_it_work import DidItWork

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s',
)
_L = logging.getLogger('experiment')

VARIANTS = ['v1', 'v2', 'v3', 'v4']
SHOTS = [0, 3, 8]
SEED = 42
_OUTPUT = Path(os.environ.get('RESULTS_DIR', 'results/'))


def go(model_key, quantize='8bit'):
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    ts = DT.now().strftime('%Y%m%d_%H%M%S')

    dp = Path('data/test_set.json')
    gen = BigGenerator(seed=SEED)
    if dp.exists():
        _L.info('Loading existing test set from ' + str(dp))
        with open(dp) as _f:
            _raw = json.load(_f)
        test_set = gen.generate_test_set()
    else:
        _L.info('Generating test data...')
        test_set = gen.save_dataset()

    _L.info(f'Loading {model_key} ({quantize})...')
    model = Brain(model_key, quantize=quantize)
    _L.info('VRAM usage: ' + model.get_vram_usage())

    all_results = []
    total = len(VARIANTS) * len(SHOTS) * 70
    _L.info(
        f'Running {len(VARIANTS)} variants x {len(SHOTS)} shots x 70 sentences = {total} instances'
    )

    for v in VARIANTS:
        builder = Promptificator(v)
        checker = DidItWork(v)
        examples_pool = gen.generate_few_shot_examples(v, n=8)

        for ns in SHOTS:
            ex = examples_pool[:ns]
            _L.info(f'Running: {model_key} | {v} | {ns}-shot')

            _counter = 0
            for sent in tqdm(test_set, desc=v + '/' + str(ns) + '-shot'):
                if model.model_type == 'instruction':
                    prompt = builder.build_chat_prompt(
                        sent.english,
                        ex if ns > 0 else None,
                    )
                else:
                    prompt = builder.build_prompt(
                        sent.english,
                        ex if ns > 0 else None,
                    )

                prediction = model.generate(prompt)
                gold = getattr(sent, v)
                result = checker.evaluate_single(prediction, gold)

                _row = {}
                _row['model'] = model_key
                _row['variant'] = v
                _row['shot_count'] = ns
                _row['sentence_id'] = _counter
                _row['english'] = sent.english
                _row['gold'] = gold
                _row['prediction'] = prediction
                _row['exact_match'] = result.exact_match
                _row['edit_distance'] = result.edit_distance
                _row['word_order_correct'] = result.word_order_correct
                _row['case_marking_correct'] = result.case_marking_correct
                all_results.append(_row)
                _counter = _counter + 1

    df = pandas.DataFrame(all_results)
    csv_path = _OUTPUT / ('results_' + model_key + '_' + ts + '.csv')
    df.to_csv(csv_path, index=False)
    _L.info(f'Results saved to {csv_path}')

    _L.info('=' * 50)
    _L.info(model_key.upper() + ' COMPLETE')
    _L.info('Instances: ' + str(len(df)))
    _L.info('Accuracy: ' + str(round(df['exact_match'].mean(), 3)))
    _L.info('=' * 50)
    return csv_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment for one model')
    parser.add_argument('--model', required=True, choices=['pythia', 'bloomz', 'smol'], help='Which model to run')
    parser.add_argument('--quantize', default='8bit', choices=['8bit', '4bit', 'none'], help='Quantisation level (default: 8bit)')
    args = parser.parse_args()
    go(args.model, args.quantize)
