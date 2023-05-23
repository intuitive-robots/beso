from typing import Optional, Callable, Any

from pathlib import Path
import torch
from torch.utils.data import TensorDataset, Dataset
import numpy as np


from beso.networks.scaler.scaler_class import Scaler
from beso.envs.dataloaders.trajectory_loader import TrajectoryDataset, get_train_val_sliced, split_traj_datasets
from beso.envs.utils import transpose_batch_timestep



class RelayKitchenTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(self, data_directory, device="cpu", onehot_goals=False):
        data_directory = Path(data_directory)
        observations = torch.from_numpy(
            np.load(data_directory / "observations_seq.npy")
        )[:, :, :30] # only get ninzero stuff
        actions = torch.from_numpy(np.load(data_directory / "actions_seq.npy"))
        masks = torch.from_numpy(np.load(data_directory / "existence_mask.npy"))
        goals = torch.load(data_directory / "onehot_goals.pth")
        # The current values are in shape T x N x Dim, move to N x T x Dim
        observations, actions, masks, goals = transpose_batch_timestep(
            observations, actions, masks, goals
        )
        self.masks = masks
        tensors = [observations, actions, masks]
        if onehot_goals:
            tensors.append(goals)
        tensors = [t.to(device).float() for t in tensors]
        TensorDataset.__init__(self, *tensors)
        self.actions = self.tensors[1]
        self.observations = self.tensors[0]
        self.onehot_goals = goals

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum())
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


def get_relay_kitchen_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    only_sample_tail: bool = False,
    only_sample_seq_end: bool = False,
    transform: Optional[Callable[[Any], Any]] = None,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]
    return get_train_val_sliced(
        RelayKitchenTrajectoryDataset(
            data_directory, onehot_goals=(goal_conditional == "onehot")
        ),
        train_fraction,
        random_seed,
        device,
        window_size,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
        transform=transform,
        only_sample_tail=only_sample_tail,
        only_sample_seq_end=only_sample_seq_end,
    )


class RelayKitchenVisionTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(self, data_directory, device="cpu", onehot_goals=False):
        data_directory = Path(data_directory)
        states = torch.from_numpy(np.load(data_directory / "observations_seq.npy"))
        actions = torch.from_numpy(np.load(data_directory / "actions_seq.npy"))
        masks = torch.from_numpy(np.load(data_directory / "existence_mask.npy"))
        goals = torch.load(data_directory / "onehot_goals.pth")
        # The current values are in shape T x N x Dim, move to N x T x Dim
        states, actions, masks, goals = transpose_batch_timestep(
            states, actions, masks, goals
        )
        # only take the joint angles (7 DoF), ignore grippers and env state
        states = states[:, :, :7]
        imgs_embedding = torch.load(data_directory / "observations_seq_embedding.pth")
        observations = torch.cat([imgs_embedding, states], dim=2)
        self.masks = masks
        tensors = [observations, actions, masks]
        if onehot_goals:
            tensors.append(goals)
        tensors = [t.to(device).float() for t in tensors]
        TensorDataset.__init__(self, *tensors)
        self.actions = self.tensors[1]

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)



def get_relay_kitchen_vision_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    only_sample_tail: bool = False,
    only_sample_seq_end: bool = False,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]
    return get_train_val_sliced(
        RelayKitchenVisionTrajectoryDataset(
            data_directory, onehot_goals=(goal_conditional == "onehot")
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
    )
