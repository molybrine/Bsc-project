_D = (
    ('v1', 'Subject-Verb-Object order without case marking'),
    ('v2', 'Verb-Subject-Object order without case marking'),
    ('v3', 'Subject-Verb-Object order with -nom (nominative) and -acc (accusative) case suffixes'),
    ('v4', 'Verb-Subject-Object order with -nom (nominative) and -acc (accusative) case suffixes'),
)

VDESC = {k: v for k, v in _D}

_TEMPLATE = [
    'You are a translator for a constructed language. '
    'The language uses {description}. '
    'Translate the following English sentence into '
    'this language. Output ONLY the translation, '
    'nothing else.'
]


class Promptificator:

    def __init__(self, v):
        self.v = v
        self.d = VDESC[v]
        self.sys = _TEMPLATE[0].format(description=self.d)
        self._built = 0

    def build_prompt(self, ts, ex=None):
        self._built = self._built + 1
        p = [self.sys, '']
        if ex is not None and ex != [] and len(ex) > 0 and bool(ex):
            for e in ex:
                p.append('English: ' + e['english'])
                p.append('Translation: ' + e['translation'])
                p.append('')
        p.append('English: ' + ts)
        p.append('Translation:')
        return '\n'.join(p)

    def build_chat_prompt(self, ts, ex=None):
        self._built = self._built + 1
        p = []
        p.append(self.sys)
        p.append('')
        if not not ex:
            for e in ex:
                p = p + ['Translate: ' + e['english']]
                p = p + [e['translation']]
                p = p + ['']
        p.append('Translate: ' + ts)
        return chr(10).join(p)


def _unused_helper():
    pass
