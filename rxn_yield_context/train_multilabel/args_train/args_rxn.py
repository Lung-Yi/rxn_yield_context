# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 20:52:09 2020

@author: Lung-Yi

args for Reaction classification model
"""
import json
import os

import pickle
from typing import List, Optional, Tuple
from typing_extensions import Literal

import torch
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)


def get_checkpoint_paths(checkpoint_path: Optional[str] = None,
                         checkpoint_paths: Optional[List[str]] = None,
                         checkpoint_dir: Optional[str] = None,
                         ext: str = '.pt') -> Optional[List[str]]:
    """
    Gets a list of checkpoint paths either from a single checkpoint path or from a directory of checkpoints.

    If :code:`checkpoint_path` is provided, only collects that one checkpoint.
    If :code:`checkpoint_paths` is provided, collects all of the provided checkpoints.
    If :code:`checkpoint_dir` is provided, walks the directory and collects all checkpoints.
    A checkpoint is any file ending in the extension ext.

    :param checkpoint_path: Path to a checkpoint.
    :param checkpoint_paths: List of paths to checkpoints.
    :param checkpoint_dir: Path to a directory containing checkpoints.
    :param ext: The extension which defines a checkpoint file.
    :return: A list of paths to checkpoints or None if no checkpoint path(s)/dir are provided.
    """
    if sum(var is not None for var in [checkpoint_dir, checkpoint_path, checkpoint_paths]) > 1:
        raise ValueError('Can only specify one of checkpoint_dir, checkpoint_path, and checkpoint_paths')

    if checkpoint_path is not None:
        return [checkpoint_path]

    if checkpoint_paths is not None:
        return checkpoint_paths

    if checkpoint_dir is not None:
        checkpoint_paths = []

        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(ext):
                    checkpoint_paths.append(os.path.join(root, fname))

        if len(checkpoint_paths) == 0:
            raise ValueError(f'Failed to find any checkpoints with extension "{ext}" in directory "{checkpoint_dir}"')

        return checkpoint_paths

    return None

class TrainArgs_rxn(Tap):
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""

    # General arguments
    # data_path: str
    # """Path to data CSV file."""
    dataset_type: Literal['regression', 'classification', 'multiclass','multilabel'] = 'multilabel'
    """Type of dataset. This determines the loss function used during training."""
    multilabel_num_classes: int = 0
    """Number of classes when running multilabel classification. (default value is for reagent prediction) """
    reagent_num_classes: int = 0
    """Number of classes when running multilabel classification.(*only in Multi-task ) """
    solvent_num_classes: int = 0
    """Number of classes when running multilabel classification.(*only in Multi-task ) """

    seed: int = 0
    """
    Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.
    """
    pytorch_seed: int = 0
    """Seed for PyTorch randomness (e.g., random initial weights)."""
    metric: Literal['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy'] = None
    """Metric to use during evaluation. Defaults to "auc" for classification and "rmse" for regression."""
    save_dir: str = None # TODO: must input this argument
    """Directory where model checkpoints will be saved."""
    second_part_data: str = None
    """The path of second part data (fake data augmentation) """
    val_path: str = None
    """validation data path """
    train_path: str = None
    """ training data path """


    # Model arguments
    fpsize: int = 4096
    """feature length of morgan fingerprint """
    radius: int = 2
    """feature radius of morgan fingerprint """
    fp_type: str = 'morgan'
    """fingerprint type: ['morgan', 'drfp'] """
    dropout: float = 0.5
    """Dropout probability in Linear Layer."""
    activation: Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'
    """Activation function."""
    # Model arguments only for first part:
    loss: Literal['Focal', 'Asym'] = 'Focal'
    """ loss function for multilabel"""
    alpha: float = None
    """ weighing factor in focal loss"""
    gamma: float = 2
    """modulating factor in focal loss"""

    hidden_share_size: int = 512
    """Dimensionality of hidden layers in shared layer."""
    hidden_reagent_size: int = 300
    """Dimensionality of hidden layers in reagent layer."""
    hidden_solvent_size: int = 100
    """Dimensionality of hidden layers in solvent layer."""
    
    # Model arguments only for second part:
    last_output_layer_pointwise: Literal['relu', 'sigmoid'] = 'sigmoid'
    """Last output layer activation function only for the prediction of pointwise relevance. (scoring function) """
    h1_size_rxn_fp: int = 800
    """First hidden layer size for encoding reaction fp. (size=32768) """
    h_size_solvent: int = 50
    """Hidden layer size for encoding one-hot solvent classes. (size = number of solvent classes) """
    h_size_reagent: int = 100
    """Hidden layer size for encoding one-hot reagent classes. (size = number of reagent classes) """
    h2_size: int = 300
    """Second hidden layer for encoding total rxn_fp_sol_reag """
    num_last_layer: int = 1
    """number of last hidden layer """
    num_shared_layer: int = 1
    """number of shared layer """
    cutoff_solv: float = 0.15
    """ cutoff of solvent """
    cutoff_reag: float = 0.2
    """ cutoff of reagent """
    
    # Training arguments
    valid_per_epoch: int = 1
    """ Validate the model per epoch """
    redo_epoch: int = 5
    """ Redo augmentation every how many epoch"""
    num_fold: int = 7
    """Data augmentation number of fold """
    train_info: bool = False
    """Whether to calculate training error and accuracy """
    epochs: int = 30
    """Number of epochs to run."""
    warmup_epochs: float = 2.0
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    init_lr: float = 1e-4
    """Initial learning rate."""
    max_lr: float = 1e-3
    """Maximum learning rate."""
    final_lr: float = 1e-4
    """Final learning rate."""
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""
    checkpoint_dir: str = None
    """Directory from which to load model checkpoints (walks directory and ensembles all models that are found)."""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pt` file)."""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pt` files)."""
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    features_generator: List[str] = None
    """Method(s) of generating additional features."""
    features_path: List[str] = None
    """Path(s) to features to use in FNN (instead of features_generator)."""
    no_features_scaling: bool = False
    """Turn off scaling of features."""
    max_data_size: int = None
    """Maximum number of data points to load."""
    num_workers: int = 0
    """Number of workers for the parallel data loading (0 means sequential)."""
    batch_size: int = 50
    """Batch size."""
    weight_decay: float = 0.
    """weight decay for optimizer """



    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs_rxn, self).__init__(*args, **kwargs)
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = ['reagents']
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None



    @property
    def num_lrs(self) -> int:
        """The number of learning rates to use (currently hard-coded to 1)."""
        return 1


    @property
    def train_data_size(self) -> int:
        """The size of the training data set."""
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size
        
    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs) or not."""
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda
    def process_args(self) -> None:
        super(TrainArgs_rxn, self).process_args()


        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
        )


        # Process and validate metric and loss function
        if self.metric is None:
            if self.dataset_type == 'classification':
                self.metric = 'auc'
            elif self.dataset_type == 'multiclass':
                self.metric = 'cross_entropy'
            elif self.dataset_type == 'multilabel':
                self.metric = 'cross_entropy'
            else:
                self.metric = 'rmse'


