"""
Module 3: Model Inference Pipeline
Loads quantised models and generates translations.
Supports NVIDIA (CUDA), AMD (ROCm), Apple (MPS), and CPU fallback.
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# Model registry
MODEL_CONFIGS = {
    'pythia': {
        'name': 'EleutherAI/pythia-6.9b',
        'type': 'base',           # completion-style
        'max_new_tokens': 64,
    },
    'bloomz': {
        'name': 'bigscience/bloomz-7b1',
        'type': 'instruction',    # chat-style
        'max_new_tokens': 64,
    },
    'smol': {
        'name': 'EleutherAI/pythia-70m',
        'type': 'base',
        'max_new_tokens': 32,
    },
}


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        logger.info('Found NVIDIA GPU, using CUDA')
        return 'cuda'
    try:
        if hasattr(torch, 'hip') or torch.version.hip:
            logger.info('Found AMD GPU, using ROCm')
            return 'cuda'
    except Exception:
        pass
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info('Found Apple GPU, using MPS')
            return 'mps'
    except Exception:
        pass
    logger.info('No GPU found, falling back to CPU (this will be slow)')
    return 'cpu'


class ModelInference:
    """Handles model loading and text generation."""

    def __init__(
        self,
        model_key: str,
        quantize: str = '8bit',
        device: Optional[str] = None,
    ):
        cfg = MODEL_CONFIGS[model_key]
        self.model_name = cfg['name']
        self.model_type = cfg['type']
        self.max_new_tokens = cfg['max_new_tokens']
        self._total_generated = 0

        detected = _detect_device() if device is None else device

        # CPU cannot run quantised models
        if detected == 'cpu' and quantize != 'none':
            logger.warning(
                'No GPU detected, forcing quantize=none for CPU'
            )
            quantize = 'none'

        # Quantisation config
        quant_config = None
        if quantize == '8bit':
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantize == '4bit':
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        if quantize == 'none':
            self.device = 'cpu'
        else:
            self.device = detected

        logger.info(
            f'Loading {self.model_name} '
            f'({quantize} quantisation)...'
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        if quantize == 'none':
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
            )
            self.model.to(self.device)
        else:
            # Auto-detect VRAM and use ~85% for model,
            # overflow the rest to CPU RAM
            total_vram = torch.cuda.get_device_properties(
                0
            ).total_memory / (1024 ** 3)
            gpu_budget = f'{int(total_vram * 0.85)}GiB'
            logger.info(
                f'VRAM detected: {total_vram:.1f} GB '
                f'— reserving {gpu_budget} for model'
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map='auto',
                max_memory={
                    0: gpu_budget, 'cpu': '16GiB'
                },
            )
        self.model.eval()
        logger.info('Model loaded successfully.')

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """Generate a single translation from a prompt."""
        self._total_generated += 1
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1024,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,        # greedy decoding
            temperature=1.0,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only newly generated tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        raw = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )

        # Extract first line (the translation)
        translation = raw.strip().split('\n')[0].strip()
        return translation

    def generate_batch(
        self, prompts: List[str]
    ) -> List[str]:
        """Generate translations for a list of prompts."""
        return [self.generate(p) for p in prompts]

    def get_vram_usage(self) -> str:
        """Report current GPU memory usage."""
        try:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(
                    0).total_memory / 1e9
                return f'{alloc:.1f} / {total:.1f} GB'
            return 'CUDA not available'
        except Exception:
            return 'Unable to read VRAM usage'
