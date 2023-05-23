import logging
import os 

from omegaconf import DictConfig
import numpy as np
import torch
import hydra
from tqdm import tqdm
import wandb

from beso.workspaces.base_workspace_manager import BaseWorkspaceManger
from beso.networks.scaler.scaler_class import MinMaxScaler
from beso.envs.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from beso.envs.utils import get_split_idx
from beso.envs.block_pushing.data.dataloader import PushTrajectoryDataset
from beso.agents.diffusion_agents.beso_agent import BesoAgent

log = logging.getLogger(__name__)


class BlockPushingManager(BaseWorkspaceManger):
    def __init__(
            self,
            seed: int,
            device: str,
            dataset_fn: DictConfig,
            goal_fn: DictConfig,
            eval_n_times: int,
            eval_n_steps,
            scale_data: bool,
            render: bool,
            train_batch_size: int = 256,
            test_batch_size: int = 256,
            num_workers: int = 4,
            train_fraction: float = 0.95,
            use_minmax_scaler: bool = False,
    ):
        super().__int__(seed, device)
        self.eval_n_times = eval_n_times
        self.eval_n_steps = eval_n_steps
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.datasets = hydra.utils.call(dataset_fn)
        self.train_set, self.test_set = self.datasets
        self.num_workers = num_workers
        self.train_fraction = train_fraction
        self.scale_data = scale_data
        self.use_minmax_scaler = use_minmax_scaler
        self.scaler = None
        self.data_loader = self.make_dataloaders()
        self.render = render
        # get goal function for evaluation
        self.goals_fn = hydra.utils.call(goal_fn)
        self.mask_obs = dataset_fn.transform.mask_targets if hasattr(dataset_fn.transform, 'mask_targets') else False
        self.reduce_obs_dim = dataset_fn.transform.reduce_obs_dim if hasattr(dataset_fn.transform, 'reduce_obs_dim') else False
        self.push_traj = PushTrajectoryDataset(
            goal_fn.data_path, onehot_goals=True
        )
        self.goal_conditional = dataset_fn.goal_conditional

    def make_dataloaders(self):
        """
        Create a training and test dataloader using the dataset instances of the task
        """
        if self.use_minmax_scaler:
            self.scaler = MinMaxScaler(self.train_set.dataset.dataset.get_all_observations(), self.train_set.dataset.dataset.get_all_actions(), self.scale_data, self.device)
        else:
            self.scaler = Scaler(self.train_set.dataset.dataset.get_all_observations(), self.train_set.dataset.dataset.get_all_actions(), self.scale_data, self.device)
        
        train_dataloader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        test_dataloader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return {
            "train": train_dataloader,
            "test": test_dataloader
        }

    def test_agent(
        self, 
        agent, 
        evaluate_multigoal: bool = True, # just for for same input as kitchen environment
        evaluate_sequential: bool = True, # just for for same input as kitchen environment
        log_wandb: bool = True, 
        new_sampler_type=None,
        n_inference_steps=None,
        get_mean=None,
        extra_args=None, 
        noise_scheduler=None,
        store_video=False,
        video_path=None  # Path to store the generated video
        ):
        """
        Test the agent on the environment with the given goal function
        """
        if store_video:
            import imageio
        
        self.env = BlockPushMultimodal(render=self.render)
        log.info('Starting trained model evaluation on the multimodal blockpush environment')
        rewards = []
        results = []
        frames = [] 
        for goal_idx in range(self.eval_n_times):
            total_reward = 0
            info = None
            done = False
            obs = self.env.reset()
            obs_res = np.hstack([np.array(v) for v in list(obs.values())])
            if goal_idx >= 950:
                goal_idx_2 = goal_idx - 950 
            else:
                goal_idx_2 = goal_idx
            goal = self.goals_fn(obs_res, goal_idx_2, 0)
            if self.reduce_obs_dim:
                goal = goal[:, :10]

            # now run the agent for n steps 
            for n in tqdm(range(self.eval_n_steps)):
                if self.render:
                    self.env.render(mode="human")
                if store_video:
                    frame = self.env.render(mode="rgb_array")
                    frames.append(frame)
                if done or n == self.eval_n_steps-1:
                    rewards.append(total_reward)
                    print('Total reward: {}'.format(total_reward))
                    result = self._report_result_upon_completion(goal_idx)
                    log.info(f"Result: {result}")
                    if log_wandb:
                        wandb.log({'Result': result,
                                   'Reward': total_reward
                                })
                    results.append(result)
                    break
                obs = np.hstack([np.array(v) for v in list(obs.values())])
                obs = torch.from_numpy(obs.reshape(1, len(obs))).to(torch.float32)
                if self.reduce_obs_dim:
                    obs = obs[:, :10]
                if self.mask_obs:
                    if self.reduce_obs_dim:
                        pass
                    else:
                        obs[:, 10:] = 0
                if isinstance(agent, BesoAgent):
                    pred_action = agent.predict(
                        {'observation': obs,
                         'goal_observation': goal}, 
                        new_sampler_type=new_sampler_type,
                        new_sampling_steps=n_inference_steps,
                        get_mean=get_mean,
                        extra_args={}, 
                        noise_scheduler=noise_scheduler,
                    )
                else:
                    sampler_dict = {}
                    if n_inference_steps is not None:
                        sampler_dict['num_sampling_steps'] = n_inference_steps 
                    pred_action = agent.predict(
                        {'observation': obs,
                         'goal_observation': goal}, 
                    )
                obs, reward, done, _ = self.env.step(pred_action.reshape(-1).detach().cpu().numpy())
                total_reward += reward
                # get current goal
                if self.goal_conditional == "onehot":
                    goal = self.goals_fn(obs, goal_idx, n)
                
        log.info(f"Total reward: {total_reward}")
        
        self.env.close()
        if store_video:
            video_filename = f"rollout_{goal_idx}.mp4"
            video_filepath = os.path.join(video_path, video_filename)

            # Save the frames as a video using imageio
            imageio.mimsave(video_filepath, frames, fps=30)

        avrg_reward = sum(rewards)/len(rewards)
        std_reward = np.array(rewards).std()
        avrg_result = sum(results)/len(results)
        std_result = np.array(results).std()
        
        print('average reward {}'.format(avrg_reward))
        print('average result {}'.format(avrg_result))
        print({"Cond_success_ratio": avrg_result/avrg_reward})
        log.info(f"Average reward: {avrg_reward} std: {std_reward}")
        log.info(f"Average result: {avrg_result} std: {std_result}")
        log.info('... finished trained model evaluation of the blockpush environment environment.')
        if log_wandb:
            wandb.log({"Average_reward": avrg_reward})
            wandb.log({"Average_result": avrg_result})
            wandb.log({"Cond_success_ratio": avrg_result/avrg_reward})
        if log_wandb:
            log.info(f"---------------------------------------")
        else:
            print("---------------------------------------")
        return_dict = {
            'avrg_reward': avrg_reward,
            'std_reward': std_reward,
            'avrg_result': avrg_result,
            'std_result': std_result
        }
        # return the average reward 
        return return_dict
    
    def _report_result_upon_completion(self, goal_idx=None):
        """
        Report the result upon completion of the episode
        """
        if goal_idx is not None:
            train_idx, val_idx = get_split_idx(
                len(self.push_traj),
                seed=self.seed,
                train_fraction=self.train_fraction,
            )
            _, _, _, onehot_goals = self.push_traj[train_idx[goal_idx]]
            onehot_mask, first_frame = onehot_goals.max(0)
            goals = [(first_frame[i], i) for i in range(4) if onehot_mask[i]]
            goals = sorted(goals, key=lambda x: x[0])
            goals = [g[1] for g in goals]
            logging.info(f"Expected tasks {goals}")
            expected_tasks = set(goals)
            conditional_done = set(self.env.all_completions).intersection(
                expected_tasks
            )
            return len(conditional_done) / 2
        else:
            return len(self.env.all_completions) / 2
