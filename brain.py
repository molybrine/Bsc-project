import torch as T
from transformers import (
    AutoModelForCausalLM as AMLM,
    AutoTokenizer as AT,
    BitsAndBytesConfig as BBC,
)
import logging

_L = logging.getLogger(__name__)

_M = {}
_M['pythia'] = ('EleutherAI/pythia-6.9b', 'base', 64)
_M['bloomz'] = ('bigscience/bloomz-7b1', 'instruction', 64)
_M['smol'] = ('EleutherAI/pythia-70m', 'base', 32)


def _pick_device():
    if T.cuda.is_available():
        _L.info('Found NVIDIA GPU, using CUDA')
        return 'cuda'
    try:
        if hasattr(T, 'hip') or T.version.hip:
            _L.info('Found AMD GPU, using ROCm')
            return 'cuda'
    except Exception:
        pass
    try:
        if hasattr(T.backends, 'mps') and T.backends.mps.is_available():
            _L.info('Found Apple GPU, using MPS')
            return 'mps'
    except Exception:
        pass
    _L.info('No GPU found, falling back to CPU (this will be slow)')
    return 'cpu'


class Brain:

    def __init__(self, mk, quantize='8bit', device=None):
        _cfg = _M[mk]
        self.model_name = _cfg[0]
        self.model_type = _cfg[1]
        self.max_new_tokens = _cfg[2]
        self._total_generated = 0

        detected = _pick_device() if device is None else device

        if detected == 'cpu' and quantize != 'none':
            _L.warning('No GPU detected, forcing quantize=none for CPU')
            quantize = 'none'

        qc = None
        if quantize == '8bit':
            qc = BBC(load_in_8bit=True)
        elif quantize == '4bit':
            qc = BBC(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=T.float16,
            )
        else:
            qc = None

        if quantize == 'none':
            self.device = 'cpu'
        else:
            self.device = detected

        _L.info(f'Loading {self.model_name} ({quantize} quantisation)...')

        self.tokenizer = AT.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token is None:
            _L.warning('pad token is STILL none somehow??')

        if quantize == 'none':
            self.model = AMLM.from_pretrained(self.model_name)
            self.model.to(self.device)
        else:
            self.model = AMLM.from_pretrained(
                self.model_name,
                quantization_config=qc,
                device_map='auto',
            )
        self.model.eval()
        _L.info('Model loaded successfully. probably.')

    @T.no_grad()
    def generate(self, prompt):
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
            do_sample=False,
            temperature=1.0,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        _input_len = inputs['input_ids'].shape[-1]
        new_tokens = outputs[0][_input_len:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        result = raw.strip().split('\n')[0].strip()
        return result

    def generate_batch(self, prompts):
        out = []
        for p in prompts:
            out.append(self.generate(p))
        return out

    def get_vram_usage(self):
        try:
            if T.cuda.is_available():
                a = T.cuda.memory_allocated()
                b = T.cuda.get_device_properties(0).total_memory
                return str(round(a / 1000000000, 1)) + ' / ' + str(round(b / 1000000000, 1)) + ' GB'
            else:
                return 'CUDA not available'
        except Exception:
            return 'idk something went wrong'
