
import logging
from typing import Optional

import numpy as np
import torch

from beso.envs.franka_kitchen.dataloader import RelayKitchenTrajectoryDataset
from beso.envs.utils import get_split_idx

'''
code adopted from https://github.com/jeffacce/play-to-policy/blob/master/envs/kitchen/goals.py
'''


def rearrange_array(a1, a2):
        """
        Rearranges the elements of `a1` based on the sorting order of `a2`.

        Args:
            a1 (List or ndarray): The array to be rearranged.
            a2 (List or ndarray): The array used for sorting.

        Returns:
            List: The elements of `a1` in the sorted order of `a2`.
        """
        sorted_indices = sorted(range(len(a2)), key=lambda k: a2[k])
        return [a1[i] for i in sorted_indices]


def get_goal_fn(
    data_path,
    goal_conditional: Optional[str] = None,
    goal_seq_len: Optional[int] = None,
    sequential_goal: Optional[bool] = None,
    seed: Optional[int] = None,
    train_fraction: Optional[float] = None,
    device: str = 'cuda'
):
    """
    Returns a goal function based on the specified conditions.

    Args:
        data_path (str): The path to the data.
        goal_conditional (str, optional): The type of goal conditioning ("future" or "onehot").
        goal_seq_len (int, optional): The length of the goal sequence.
        sequential_goal (bool, optional): Whether the goals are sequential or not.
        seed (int, optional): The random seed.
        train_fraction (float, optional): The fraction of data used for training.
        device (str, optional): The device used for computations.

    Returns:
        function: The goal function.

    Raises:
        AssertionError: If `goal_seq_len` is not provided for "future" goal conditioning.

    Notes:
        - If `goal_conditional` is None, the goal function returns None.
        - If `goal_conditional` is "future" and `sequential_goal` is False or None, the goal function returns the last `goal_seq_len` observations from the dataset.
        - If `goal_conditional` is "future" and `sequential_goal` is True, the goal function returns the goal sequence starting from the specified `goal_idx` and `goal_number`. 
          It also returns the task name associated with the goal.
        - If `goal_conditional` is "onehot", the goal function returns the one-hot encoded goal corresponding to the `frame_idx`.
    """
    relay_traj = RelayKitchenTrajectoryDataset(
        data_path, device=device, onehot_goals=True
    )
    train_idx, val_idx = get_split_idx(
        len(relay_traj),
        seed=seed,
        train_fraction=train_fraction,
    )
    all_tasks = np.array(
            [
                'bottom burner', 'top burner', 'light switch', 'slide cabinet',
                'hinge cabinet', 'microwave', 'kettle'
            ],
            dtype='<U13'
        )
    if goal_conditional is None:
        goal_fn = lambda state: None
    elif goal_conditional == "future" and sequential_goal is False or sequential_goal is None:
        assert (
            goal_seq_len is not None
        ), "goal_seq_len must be provided if goal_conditional is 'future'"

        def goal_fn(state, goal_idx, frame_idx):
            if goal_idx > 555:
                goal_idx = goal_idx - 555
            logging.info(f"goal_idx: {train_idx[goal_idx]}")
            obs, _, _, _ = relay_traj[train_idx[goal_idx]]  # seq_len x obs_dim
            obs = obs[-goal_seq_len:]
            return obs
    
    elif goal_conditional == "future" and sequential_goal is True:
        assert (
            goal_seq_len is not None
        ), "goal_seq_len must be provided if goal_conditional is 'future'"

        def goal_fn(state, goal_idx, goal_number=1):
            if goal_number == 0:
                goal_number = 1
                print("goal_number is 0, setting to 1")
            if goal_idx > 555:
                goal_idx = goal_idx - 555
            logging.info(f"goal_idx: {train_idx[goal_idx]}")
            obs, _, _, onehot_goals = relay_traj[train_idx[goal_idx]]  # seq_len x obs_dim

            expected_mask = onehot_goals.max(0).values.bool().detach().cpu().numpy()
            order = onehot_goals.max(0).indices.detach().cpu().numpy()[expected_mask]
            order.sort()
            goal_index = order[goal_number] if len(order) > goal_number else order[-1]
            if goal_number == 4:
                task_name = all_tasks[onehot_goals[min(goal_index+5, len(onehot_goals) - 1)].bool().detach().cpu().numpy()].item()
                goal_index = 280
                return obs[-goal_seq_len:], goal_index, task_name
            # task_idx = np.where(onehot_goals.max(0).indices == goal_index)
            expected_tasks = all_tasks[expected_mask]
            expected_mask = rearrange_array(expected_tasks, onehot_goals.max(0).indices.detach().cpu().numpy()[expected_mask])
            task_name = all_tasks[onehot_goals[min(goal_index-1, len(onehot_goals) - 1)].bool().detach().cpu().numpy()].item()
            return obs[goal_index:(goal_index+goal_seq_len)], goal_index, task_name # [order.index(goal_idx)].reshape(4, 7) ]

    elif goal_conditional == "onehot":

        def goal_fn(state, goal_idx, frame_idx):
            if frame_idx == 0:
                logging.info(f"goal_idx: {train_idx[goal_idx]}")
            _, _, _, onehot_goals = relay_traj[train_idx[goal_idx]]  # seq_len x obs_dim
            
            return onehot_goals[min(frame_idx, len(onehot_goals) - 1)].reshape(1, 7)

    return goal_fn

