import torch
import logging
import numpy as np
from typing import Optional
from beso.envs.block_pushing.data.dataloader import PushTrajectoryDataset


# code adopted from play-to-policy

def get_split_idx(l, seed, train_fraction=0.95):
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(l, generator=rng).tolist()
    l_train = int(l * train_fraction)
    return idx[:l_train], idx[l_train:]



def get_goal_fn(
    data_path,
    goal_conditional: Optional[str] = None,
    goal_seq_len: Optional[int] = None,
    seed: Optional[int] = None,
    train_fraction: Optional[float] = None,
    zero_goals: Optional[bool] = True,
):
    """
    Returns a goal function based on the specified conditions.

    Args:
        data_path (str): The path to the data.
        goal_conditional (str, optional): The type of goal conditioning ("future" or "onehot").
        goal_seq_len (int, optional): The length of the goal sequence.
        seed (int, optional): The random seed.
        train_fraction (float, optional): The fraction of data used for training.
        zero_goals (bool, optional): Whether to zero out the goals or not.

    Returns:
        function: The goal function.

    Notes:
        - If `goal_conditional` is None, the goal function returns None.
        - If `goal_conditional` is "future", the goal function returns the last `goal_seq_len` observations from the dataset. 
          If `zero_goals` is True, the goals are zeroed out.
        - If `goal_conditional` is "onehot", the goal function returns a one-hot encoded goal based on the `frame_idx`. 
          If `zero_goals` is True, the goals are zeroed out except for the last completed goal.
    """
    push_traj = PushTrajectoryDataset(
        data_path, onehot_goals=True
    )
    train_idx, val_idx = get_split_idx(
        len(push_traj),
        seed=seed,
        train_fraction=train_fraction,
    )
    if goal_conditional is None:
        goal_fn = lambda state: None
        
    elif goal_conditional == "future":

        assert (
            goal_seq_len is not None
        ), "goal_seq_len must be provided if goal_conditional is 'future'"

        def goal_fn(state, goal_idx, frame_idx):
            # assuming at this point the state hasn't been masked yet by the obs_encoder
            obs, _, _, _ = push_traj[train_idx[goal_idx]]
            obs = obs.clone()
            # bugfix: the targets spawn in two possible configurations (either red or green on top)
            # so here we need to actually look at the targets to condition on the correct positions
            block_idx = [[0, 1], [3, 4]]
            target_idx = [[10, 11], [13, 14]]
            tgt_0_pos_state = torch.Tensor(state[target_idx[0]])
            tgt_0_pos_goal = obs[-1, target_idx[0]]
            goals_flipped = (tgt_0_pos_goal - tgt_0_pos_state).norm() > 0.2
            if goals_flipped:
                temp = obs[:, block_idx[0]].clone()
                obs[:, block_idx[0]] = obs[:, block_idx[1]]
                obs[:, block_idx[1]] = temp
            if zero_goals:
                obs[..., [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = 0
            obs = obs[-1:].repeat(goal_seq_len, 1)
            return obs

    elif goal_conditional == "onehot":

        def goal_fn(state, goal_idx, frame_idx):
            _, _, _, onehot_goals = push_traj[train_idx[goal_idx]]
            onehot_mask, first_frame = onehot_goals.max(0)
            goals = [(first_frame[i], i) for i in range(4) if onehot_mask[i]]
            goals = sorted(goals, key=lambda x: x[0])
            goals = [g[1] for g in goals]
            last_goal = goals[-1]  # in case everything is done, return the last goal
            if frame_idx == 0:
                logging.info(f"goal_idx: {train_idx[goal_idx]}")
                logging.info(f"goals: {goals}")

            # determine which goals are already done
            block_idx = [[0, 1], [3, 4]]
            target_idx = [[10, 11], [13, 14]]
            close_eps = 0.05
            for b in range(2):
                for t in range(2):
                    blk = state[block_idx[b]]
                    tgt = state[target_idx[t]]
                    dist = np.linalg.norm(blk - tgt)
                    if dist < close_eps:
                        if (2 * b + t) in goals:
                            goals.remove(2 * b + t)
            result = torch.zeros(4)
            if len(goals) > 0:
                result[goals[0]] = 1
            else:
                result[last_goal] = 1
            return result

    return goal_fn