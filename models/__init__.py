import importlib
from os import path as osp

from basicsr.utils import get_root_logger, scandir

# import all the model modules

model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder)
    if (not v.endswith('__init__.py')) and (not v.endswith("_utils.py"))
]
_model_modules = [
    importlib.import_module(f'models.{file_name}')
    for file_name in model_filenames
]

def create_model(config, model_type):
    
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model = model_cls(config)
    
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
