from .model import Base, PANet

__all__ = {
    'Base': Base,
    'PANet': PANet
}

def build_network(cfg):
    return __all__[cfg.MODEL.NAME](cfg)