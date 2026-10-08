from omegaconf import OmegaConf

# Register a custom resolver to multiply
OmegaConf.register_new_resolver("mul", lambda a, b: int(float(a) * float(b)))
OmegaConf.register_new_resolver("div", lambda a, b: float(a) / float(b))
