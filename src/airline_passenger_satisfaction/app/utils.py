from pathlib import Path
from typing import Dict

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def get_catalog():
    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        context = session.load_context()
        catalog = context.catalog
    return catalog


def load_model():
    pass


def _preprocess():
    pass


def _predict():
    pass


def _postprocess():
    pass


def inference() -> Dict[str, Dict[str, str]]:
    pass
