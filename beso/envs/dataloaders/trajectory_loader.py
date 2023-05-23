import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, Subset
from pathlib import Path
import numpy as np
from typing import Union, Callable, Optional, Sequence, List, Any
from tqdm import tqdm
import abc
from torch import default_generator, randperm
from torch._utils import _accumulate

# from beso.envs.toy_task_1.multipath_dataset import MultiPathTrajectoryDataset



class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        raise NotImplementedError


class TrajectorySubset(TrajectoryDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.
    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def get_all_actions(self):
        observations, actions, masks = self.dataset[self.indices]

        result = []
        # mask out invalid actions
        for i in range(len(masks)):
            T = int(masks[i].sum().item())
            result.append(actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        observations, actions, masks = self.dataset[self.indices]

        result = []
        # mask out invalid actions
        for i in range(len(masks)):
            T = int(masks[i].sum().item())
            result.append(observations[i, :T, :])
        return torch.cat(result, dim=0)


class TrajectorySlicerDataset(TrajectoryDataset):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        future_conditional: bool = False,
        min_future_sep: int = 0,
        future_seq_len: Optional[int] = None,
        only_sample_tail: bool = False,
        only_sample_seq_end: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        Slice a trajectory dataset into unique (but overlapping) sequences of length `window`.
        dataset: a trajectory dataset that satisfies:
            dataset.get_seq_length(i) is implemented to return the length of sequence i
            dataset[i] = (observations, actions, mask)
            observations: Tensor[T, ...]
            actions: Tensor[T, ...]
            mask: Tensor[T]
                0: invalid
                1: valid
        window: int
            number of timesteps to include in each slice
        future_conditional: bool = False
            if True, observations will be augmented with future observations sampled from the same trajectory
        min_future_sep: int = 0
            minimum number of timesteps between the end of the current sequence and the start of the future sequence
            for the future conditional
        future_seq_len: Optional[int] = None
            the length of the future conditional sequence;
            required if future_conditional is True
        only_sample_tail: bool = False
            if True, only sample future sequences from the tail of the trajectory
        transform: function (observations, actions, mask[, goal]) -> (observations, actions, mask[, goal])
        """
        if future_conditional:
            assert future_seq_len is not None, "must specify a future_seq_len"
        self.dataset = dataset
        self.window = window
        self.future_conditional = future_conditional
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        # sample_tail uses the final state of the trajectory as the initial state of the future sequence
        self.only_sample_tail = only_sample_tail
        # seq end uses the last state of the small training sequence as the goal state 
        # this method is used for the latent plans model
        self.only_sample_seq_end = only_sample_seq_end
        self.transform = transform
        self.slices = []
        min_seq_length = np.inf
        for i in range(len(self.dataset)):  # type: ignore
            T = self.dataset.get_seq_length(i)  # avoid reading actual seq (slow)
            min_seq_length = min(T, min_seq_length)
            if T - window < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={window}")
            else:
                self.slices += [
                    (i, start, start + window) for start in range(T - window + 1)
                ]  # slice indices follow convention [start, end)

        if min_seq_length < window:
            print(
                f"Ignored short sequences. To include all, set window <= {min_seq_length}."
            )

    def get_seq_length(self, idx: int) -> int:
        if self.future_conditional:
            return self.future_seq_len + self.window
        else:
            return self.window
    
    def get_completed_goals(self) -> torch.Tensor:
        goals = []

    def get_all_actions(self) -> torch.Tensor:
        return self.dataset.get_all_actions()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        data_batch = {}
        i, start, end = self.slices[idx]
        values = [
            x[start:end] for x in self.dataset[i]
        ]  # [observations, actions, mask]
        data_batch['observation'] = self.dataset[i][0][start:end] 
        data_batch['action'] = self.dataset[i][1][start:end]
        
        if self.future_conditional:
            valid_start_range = (
                end + self.min_future_sep,
                self.dataset.get_seq_length(i) - self.future_seq_len,
            )
            if valid_start_range[0] < valid_start_range[1]:
                if self.only_sample_tail:
                    future_obs = self.dataset[i][0][-self.future_seq_len :]
                elif self.only_sample_seq_end:
                    future_obs = self.dataset[i][0][end : (end + self.future_seq_len)]
                else:
                    start = np.random.randint(*valid_start_range)
                    end = start + self.future_seq_len
                    future_obs = self.dataset[i][0][start:end]
            else:
                # zeros placeholder T x obs_dim
                _, obs_dim = values[0].shape
                future_obs = torch.zeros((self.future_seq_len, obs_dim))
            
            # [observations, actions, mask, future_obs (goal conditional)]
            values.append(future_obs)
            data_batch['goal_observation'] = future_obs
        
            # [observations, actions, mask, future_obs (goal conditional)]
            # values.append(future_obs)
        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)
        return data_batch # tuple(values)


def get_train_val_sliced(
    traj_dataset: TrajectoryDataset,
    train_fraction: float = 0.95,
    random_seed: int = 42,
    device: Union[str, torch.device] = "cpu",
    window_size: int = 10,
    future_conditional: bool = False,
    min_future_sep: int = 0,
    future_seq_len: Optional[int] = None,
    only_sample_tail: bool = False,
    only_sample_seq_end: bool = False,
    transform: Optional[Callable[[Any], Any]] = None,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    traj_slicer_kwargs = {
        "window": window_size,
        "future_conditional": future_conditional,
        "min_future_sep": min_future_sep,
        "future_seq_len": future_seq_len,
        "only_sample_tail": only_sample_tail,
        "only_sample_seq_end": only_sample_seq_end,
        "transform": transform,
    }
    if window_size > 0:
        train_slices = TrajectorySlicerDataset(train, **traj_slicer_kwargs)
        val_slices = TrajectorySlicerDataset(val, **traj_slicer_kwargs)
        return train_slices, val_slices
    else:
        return train, val


def random_split_traj(
    dataset: TrajectoryDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajectorySubset]:
    """
    (Modified from torch.utils.data.dataset.random_split)
    Randomly split a trajectory dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:
    >>> random_split_traj(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    Args:
        dataset (TrajectoryDataset): TrajectoryDataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        TrajectorySubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set
