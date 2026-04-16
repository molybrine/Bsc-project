"""
Module 2: Prompt Construction Engine
Builds structured prompts for each (model, variant, shot) condition.
"""
from typing import List, Dict, Optional


# Variant descriptions for the system instruction
VARIANT_DESCRIPTIONS = {
    'v1': 'Subject-Verb-Object order without case marking',
    'v2': 'Verb-Subject-Object order without case marking',
    'v3': 'Subject-Verb-Object order with -nom (nominative) '
          'and -acc (accusative) case suffixes',
    'v4': 'Verb-Subject-Object order with -nom (nominative) '
          'and -acc (accusative) case suffixes',
}


class PromptBuilder:
    """Constructs prompts for all experimental conditions."""

    SYSTEM_TEMPLATE = (
        'You are a translator for a constructed language. '
        'The language uses {description}. '
        'Translate the following English sentence into '
        'this language. Output ONLY the translation, '
        'nothing else.'
    )

    def __init__(self, variant: str):
        self.variant = variant
        self.description = VARIANT_DESCRIPTIONS[variant]
        self.system_msg = self.SYSTEM_TEMPLATE.format(
            description=self.description
        )

    def build_prompt(
        self,
        test_sentence: str,
        examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Build a complete prompt string.

        Args:
            test_sentence: English sentence to translate.
            examples: List of {'english': ..., 'translation': ...}
                      dicts for few-shot demonstrations.
        Returns:
            Formatted prompt string.
        """
        parts = [self.system_msg, '']
        if examples:
            for ex in examples:
                parts.append(
                    f"English: {ex['english']}"
                )
                parts.append(
                    f"Translation: {ex['translation']}"
                )
                parts.append('')
        parts.append(f'English: {test_sentence}')
        parts.append('Translation:')
        return '\n'.join(parts)

