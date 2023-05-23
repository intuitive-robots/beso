
import torch 
import einops 


def get_split_idx(l, seed, train_fraction=0.95):
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(l, generator=rng).tolist()
    l_train = int(l * train_fraction)
    return idx[:l_train], idx[l_train:]


def blockpush_mask_targets(
    mask_targets: bool = False,
    reduce_obs_dim: bool = False,
):  
    if mask_targets:
        if reduce_obs_dim:
            def transform(input):
                assert len(input) == 4, "Input length must be 4: (obs, act, mask, goal)"
                # assume obs, act, mask, goal
                obs = input[0].clone()
                obs[..., 10:] = 0
                goal = input[-1].clone()
                goal[..., [2, 5, 6, 7, 8, 9]] = 0
                # only get the target positions of the blocks
                '''if len(goal.shape) == 3:
                    goal = goal[:, :, [0,1 , 3, 4]]
                else:
                    goal = goal[:, [0,1 , 3, 4]]'''
                return (obs, *input[1:-1], goal)
        else:
            def transform(input):
                assert len(input) == 4, "Input length must be 4: (obs, act, mask, goal)"
                # assume obs, act, mask, goal
                obs = input[0].clone()
                obs[..., 10:] = 0
                goal = input[-1].clone()
                goal[..., [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = 0
                # only get the target positions of the blocks
                '''if len(goal.shape) == 3:
                    goal = goal[:, :, [0,1 , 3, 4]]
                else:
                    goal = goal[:, [0,1 , 3, 4]]'''
                return (obs, *input[1:-1], goal)
    else:
        if reduce_obs_dim:
            def transform(input):
                assert len(input) == 4, "Input length must be 4: (obs, act, mask, goal)"
                # assume obs, act, mask, goal
                obs = input[0].clone()
                # obs[..., 10:] = 0
                goal = input[-1].clone()
                goal[..., [2, 5, 6, 7, 8, 9]] = 0
                # only get the target positions of the blocks
                '''if len(goal.shape) == 3:
                    goal = goal[:, :, [0,1 , 3, 4]]
                else:
                    goal = goal[:, [0,1 , 3, 4]]'''
                return (obs, *input[1:-1], goal)
        else:
            def transform(input):
                assert len(input) == 4, "Input length must be 4: (obs, act, mask, goal)"
                # assume obs, act, mask, goal
                obs = input[0].clone()
                # obs[..., 10:] = 0
                goal = input[-1].clone()
                goal[..., [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = 0
                
                # only get the target positions of the blocks
                '''if len(goal.shape) == 3:
                    goal = goal[:, :, [0,1 , 3, 4]]
                else:
                    goal = goal[:, [0,1 , 3, 4]]'''
                return (obs, *input[1:-1], goal)

    return transform


def transpose_batch_timestep(*args):
    return (einops.rearrange(arg, "b t ... -> t b ...") for arg in args)
