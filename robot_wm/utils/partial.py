from functools import partial

import omegaconf


def fix_partial(f: partial):
    """Converts omegaconf partial keywords to standard python types."""
    for key in f.keywords:
        value = f.keywords[key]
        if isinstance(value, omegaconf.ListConfig):
            f.keywords[key] = list(value)
        if isinstance(value, omegaconf.DictConfig):
            f.keywords[key] = dict(value)
    return f
