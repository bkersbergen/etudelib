import logging
import warnings
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything

from etudelib.config import get_configurable_parameters
from etudelib.data import get_datamodule
from etudelib.data.utils import TestSplitMode
from etudelib.models import get_model
from etudelib.utils.callbacks import LoadModelCallback, get_callbacks
from etudelib.utils.loggers import configure_logger, get_experiment_logger