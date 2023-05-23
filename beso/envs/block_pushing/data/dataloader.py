from typing import Optional, Callable, Any
import logging

import os
import torch
from torch.utils.data import TensorDataset
from pathlib import Path
import numpy as np

from beso.networks.scaler.scaler_class import Scaler
from beso.envs.dataloaders.trajectory_loader import TrajectoryDataset, get_train_val_sliced


def get_push_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    scale_data: bool=False,
    window_size=10,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    only_sample_tail: bool = False,
    only_sample_seq_end: bool = False,
    reduce_obs_dim: Optional[bool] = False,
    transform: Optional[Callable[[Any], Any]] = None,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]

    return get_train_val_sliced(
        PushTrajectoryDataset(
            data_directory, onehot_goals=(goal_conditional == "onehot"), 
            reduce_obs_dim=reduce_obs_dim
        ),
        train_fraction,
        random_seed,
        device,
        window_size,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
        only_sample_tail=only_sample_tail,
        only_sample_seq_end=only_sample_seq_end,
        transform=transform,
    )


class PushTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        device="cpu",
        onehot_goals=False,
        reduce_obs_dim=False,
        
    ):
        self.device = device
        self.data_directory = Path(data_directory)
        logging.info("Multimodal loading: started")
        self.observations = np.load(
            self.data_directory / "multimodal_push_observations.npy"
        )
        self.actions = np.load(self.data_directory / "multimodal_push_actions.npy")
        self.masks = np.load(self.data_directory / "multimodal_push_masks.npy")
        self.observations = torch.from_numpy(self.observations).to(device).float()
        if reduce_obs_dim:
            self.observations = self.observations[:, :, :10]
        self.actions = torch.from_numpy(self.actions).to(device).float()
        self.masks = torch.from_numpy(self.masks).to(device).float()
        self.goals = (
            torch.load(self.data_directory / "onehot_goals.pth").to(device).float()
        )
        logging.info("Multimodal loading: done")
        tensors = [self.observations, self.actions, self.masks]
        if onehot_goals:
            tensors.append(self.goals)
        # The current values are in shape N x T x Dim, so all is good in the world.
        TensorDataset.__init__(self, *tensors)

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.observations[i, :T, :])
        return torch.cat(result, dim=0)

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)
    

class PushTrajectorySequenceDataset(TensorDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        device="cpu",
        scale_data: bool=False,
        onehot_goals=False,
    ):  
        self.device = device
        self.data_directory = Path(data_directory)
        logging.info("Multimodal loading: started")
        self.observations = np.load(
            self.data_directory / "multimodal_push_observations.npy"
        )
        self.actions = np.load(self.data_directory / "multimodal_push_actions.npy")
        self.masks = np.load(self.data_directory / "multimodal_push_masks.npy")
        self.observations = torch.from_numpy(self.observations).to(device).float()
        self.actions = torch.from_numpy(self.actions).to(device).float()
        self.masks = torch.from_numpy(self.masks).to(device).float()
        self.scaler = Scaler(self.observations, self.actions, scale_data, device)
        tensors = [self.observations, self.actions, self.masks]
        logging.info("Multimodal loading: done")
        # The current values are in shape N x T x Dim, so all is good in the world.
        if onehot_goals:
            tensors.append(self.goals)
        super.__init__(self, *tensors)

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)
    
    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)
    