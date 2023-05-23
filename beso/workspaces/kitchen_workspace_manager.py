import logging
from pathlib import Path
import sys
import os

from omegaconf import DictConfig
import numpy as np
import torch
import hydra
from tqdm import tqdm
import wandb
import adept_envs
import einops
import gym
import matplotlib.pyplot as plt

# from mpl_toolkits.axes_grid.inset_locator import InsetPosition
from beso.workspaces.base_workspace_manager import BaseWorkspaceManger
from beso.networks.scaler.scaler_class import Scaler
from beso.envs.utils import get_split_idx
from beso.agents.diffusion_agents.beso_agent import BesoAgent
from beso.envs.franka_kitchen.dataloader import RelayKitchenTrajectoryDataset

log = logging.getLogger(__name__)


class FrankaKitchenManager(BaseWorkspaceManger):
    def __init__(
            self,
            seed: int,
            device: str,
            dataset_fn: DictConfig,
            seq_goal_fn: DictConfig,
            multi_goal_fn: DictConfig,
            eval_n_times: int,
            eval_n_steps,
            scale_data: bool,
            render: bool,
            env_name: str,
            train_batch_size: int = 256,
            test_batch_size: int = 256,
            num_workers: int = 4,
            train_fraction: float = 0.95
    ):
        super().__int__(seed, device)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.eval_n_times = eval_n_times
        self.eval_n_steps = eval_n_steps
        self.train_batch_size = train_batch_size
        self.data_path = dataset_fn.data_directory
        self.test_batch_size = test_batch_size
        self.datasets = hydra.utils.call(dataset_fn)
        self.train_set, self.test_set = self.datasets
        self.num_workers = num_workers
        self.train_fraction = train_fraction
        self.scale_data = scale_data
        self.scaler = None
        self.data_loader = self.make_dataloaders()
        self.render = render
        self.env_name = env_name
        self.goal_conditional = dataset_fn.goal_conditional
        self.relay_traj = RelayKitchenTrajectoryDataset(
            self.data_path, onehot_goals=True
        )
        self.success_rate_1 = 0
        self.success_rate_2 = 0
        self.success_rate_3 = 0
        self.success_rate_4 = 0
        self.success_rate_5 = 0
        # get goal function for evaluation       
        self.seq_goals_fn = hydra.utils.call(seq_goal_fn)
        self.multi_goals_fn = hydra.utils.call(multi_goal_fn)
        # check multimdodality of solutions
        self.solved_tasks = {
            'n_bottom burner': 0,
            'n_top burner': 0,
            'n_kettle': 0,
            'n_light switch': 0,
            'n_slide cabinet': 0,
            'n_hinge cabinet': 0,
            'n_microwave': 0,
        }

        self.expected_tasks = {
            'n_bottom burner': 0,
            'n_top burner': 0,
            'n_kettle': 0,
            'n_light switch': 0,
            'n_slide cabinet': 0,
            'n_hinge cabinet': 0,
            'n_microwave': 0,
        }
        self.all_tasks = np.array(
            [
                'bottom burner', 'top burner', 'light switch', 'slide cabinet',
                'hinge cabinet', 'microwave', 'kettle'
            ],
            dtype='<U13'
        )
        self.used_trajectories = []
        self.traj_count = {}
        # self.return_expert_task_completion()
    
    def reset_tasks(self):
        """
        Resets the task-related attributes.
        """
        self.solved_tasks = {
            'n_bottom burner': 0,
            'n_top burner': 0,
            'n_kettle': 0,
            'n_light switch': 0,
            'n_slide cabinet': 0,
            'n_hinge cabinet': 0,
            'n_microwave': 0,
        }

        self.expected_tasks = {
            'n_bottom burner': 0,
            'n_top burner': 0,
            'n_kettle': 0,
            'n_light switch': 0,
            'n_slide cabinet': 0,
            'n_hinge cabinet': 0,
            'n_microwave': 0,
        }

        self.used_trajectories = []
        self.traj_count = {}


    def make_dataloaders(self):
        """
        Creates training and test dataloaders using the dataset instances of the task.

        Returns:
            dict: A dictionary containing the created train and test dataloaders.
        """
        self.scaler = Scaler(
            self.train_set.dataset.dataset.get_all_observations(),
            self.train_set.dataset.dataset.get_all_actions(),
            self.scale_data, self.device)

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
        evaluate_multigoal: bool = True,
        evaluate_sequential: bool = False,
        log_wandb: bool = True, 
        new_sampler_type=None, 
        n_inference_steps=None,
        get_mean=None,
        noise_scheduler=None,
        store_video=False,
        video_path=None,  # Path to store the generated video
        extra_args={},
    ):  
        """
        Tests the agent's performance on multigoal and sequential tasks.

        Args:
            agent: The agent to be tested.
            evaluate_multigoal (bool): Optional. Indicates whether to evaluate multigoal tasks. Defaults to True.
            evaluate_sequential (bool): Optional. Indicates whether to evaluate sequential tasks. Defaults to True.
            log_wandb (bool): Optional. Indicates whether to log the results using Weights & Biases. Defaults to True.
            new_sampler_type: Optional. The new sampler type to use for action sampling. Defaults to None.
            n_inference_steps: Optional. The number of inference steps. Defaults to None.
            get_mean: Optional. The number of samples to use for calculating the mean prediction. Defaults to None.
            noise_scheduler: Optional. The noise scheduler for the sigma distribution. Defaults to None.
            store_video (bool): Optional. Indicates whether to store a video of the agent's performance. Defaults to False.
            video_path: Optional. The path to store the generated video. Defaults to None.
            extra_args: Optional. Additional arguments for testing. Defaults to an empty dictionary.

        Returns:
            Tuple[Optional[Any], Optional[Any]]: A tuple containing the results of the evaluation on multigoal and sequential tasks, respectively.

        """
        mg_results = None
        seq_results = None
        if evaluate_multigoal:
            mg_results = self.test_agent_on_multigoal(agent, log_wandb, new_sampler_type, n_inference_steps, get_mean, noise_scheduler, store_video, video_path, extra_args)
        
        if evaluate_sequential:
            seq_results = self.test_agent_on_sequential_tasks(agent, log_wandb, new_sampler_type, n_inference_steps, get_mean, noise_scheduler, extra_args)
        return mg_results, seq_results
        

    def test_agent_on_multigoal(
        self, 
        agent, 
        log_wandb: bool = True, 
        new_sampler_type=None, 
        n_inference_steps=None,
        get_mean=None,
        noise_scheduler=None,
        store_video=False,
        video_path=None, # Path to store the generated video
        extra_args={}
    ):  
        """
        Tests the agent's performance on multigoal tasks.

        Args:
            agent: The agent to be tested.
            log_wandb (bool): Optional. Indicates whether to log the results using Weights & Biases. Defaults to True.
            new_sampler_type: Optional. The new sampler type to use for action sampling. Defaults to None.
            n_inference_steps: Optional. The number of inference steps. Defaults to None.
            get_mean: Optional. The number of samples to use for calculating the mean prediction. Defaults to None.
            noise_scheduler: Optional. The noise scheduler for the sigma distribution. Defaults to None.
            store_video (bool): Optional. Indicates whether to store a video of the agent's performance. Defaults to False.
            video_path: Optional. The path to store the generated video. Defaults to None.
            extra_args: Optional. Additional arguments for testing. Defaults to an empty dictionary.

        Returns:
            dict: A dictionary containing the evaluation results for the multigoal tasks.

        """
        if store_video:
            import imageio
        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)
        log.info('Starting trained model evaluation on the multimodal kitchen environment')
        rewards = []
        results = []
        frames = [] 
        for goal_idx in range(self.eval_n_times):
            if goal_idx > 536:
                goal_idx = goal_idx - 536
            total_reward = 0
            info = None
            done = False
            obs = self.env.reset()
            goal = self.multi_goals_fn(obs, goal_idx, 0)
            if isinstance(agent, BesoAgent):
                agent.reset()
            for n in tqdm(range(self.eval_n_steps)):
                if self.render:
                    self.env.render(mode="human")
                if store_video:
                    frame = self.env.render(mode="rgb_array")
                    frames.append(frame)
                if done or n == self.eval_n_steps-1:
                    rewards.append(total_reward)
                    result = self._report_result_upon_completion(goal_idx)
                    print('Total reward: {}'.format(total_reward))
                    print(f"Result: {result}")
                    if log_wandb:
                        wandb.log({'Result': result,
                                   'Reward': total_reward
                                })
                    # print(info)
                    results.append(result)
                    break
                # get current goal
                if self.goal_conditional == "onehot":
                    goal = self.multi_goals_fn(obs, goal_idx, n)
                obs = np.hstack([np.array(v) for v in list(obs)])
                # reshape and make a tensor out of the obs, we onlt need the first 30 variables
                obs = torch.from_numpy(obs.reshape(1, len(obs))).to(torch.float32)[:, :30]
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
                         'goal_observation': goal}
                    )
                
                obs, reward, done, info = self.env.step(pred_action.reshape(-1).detach().cpu().numpy())
                total_reward += reward

        log.info(f"Total reward: {total_reward}")
        
        self.env.close()
        if store_video:
            video_filename = f"rollout_{goal_idx}.mp4"
            video_filepath = os.path.join(video_path, video_filename)
            # Save the frames as a video using imageio
            imageio.mimsave(video_filepath, frames, fps=30)
        return_dict = self.compute_performance(rewards, results, log_wandb, 'multigoal')
        return return_dict
    
    def test_agent_on_sequential_tasks(
        self, 
        agent, 
        log_wandb: bool = True, 
        new_sampler_type=None, 
        n_inference_steps=None,
        get_mean=None,
        noise_scheduler=None,
        extra_args={}
    ):  
        """
        Tests the agent's performance on sequential tasks.

        Args:
            agent: The agent to be tested.
            log_wandb (bool): Optional. Indicates whether to log the results using Weights & Biases. Defaults to True.
            new_sampler_type: Optional. The new sampler type to use for action sampling. Defaults to None.
            n_inference_steps: Optional. The number of inference steps. Defaults to None.
            get_mean: Optional. The number of samples to use for calculating the mean prediction. Defaults to None.
            noise_scheduler: Optional. The noise scheduler for the sigma distribution. Defaults to None.
            extra_args: Optional. Additional arguments for testing. Defaults to an empty dictionary.

        Returns:
            dict: A dictionary containing the evaluation results for the sequential tasks.
        """
        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)
        agent.collect_plans = True
        log.info('Starting trained model evaluation on the multimodal blockpush environment')
        rewards = []
        results = []
        for goal_idx in range(self.eval_n_times):
            if goal_idx > 536:
                goal_idx = goal_idx - 536
            total_reward = 0
            info = None
            done = False
            obs = self.env.reset()
            goal_timeframe = 0
            steps = 0
            if isinstance(agent, BesoAgent):
                agent.reset()
            for goal_index in tqdm(range(1, 5, 1)):
                print(goal_index)
                prev_goal_timeframe = goal_timeframe
                goal, goal_timeframe, goal_task_name = self.seq_goals_fn(obs, goal_idx, goal_index)
                if goal_index < 4:
                    time_to_complete = goal_timeframe - prev_goal_timeframe + 50
                else:
                    time_to_complete = 280 - steps
                print('Goal: {}'.format(goal_task_name))
                delay = 0
                for n in range(time_to_complete):
                    steps += 1
                    if self.render:
                        self.env.render(mode="human")
                    if goal_task_name in self.env.all_completions and goal_index < 4:
                        print('Solved task: {}'.format(goal_task_name))
                        delay  += 1
                        if delay >10:
                            break
                        break
                    if goal_index == 4 and (done or steps == self.eval_n_steps-1 or n == time_to_complete-1):
                        rewards.append(total_reward)
                        result = self._report_result_upon_completion(goal_idx)
                        print('Total reward: {}'.format(total_reward))
                        print(f"Result: {result}")
                        if log_wandb:
                            wandb.log({'Result': result,
                                    'Reward': total_reward
                                    })
                        # print(info)
                        results.append(result)
                        break
                    obs = np.hstack([np.array(v) for v in list(obs)])
                    # reshape and make a tensor out of the obs, we onlt need the first 30 variables
                    obs = torch.from_numpy(obs.reshape(1, len(obs))).to(torch.float32)[:, :30]
                    if isinstance(agent, BesoAgent):
                        pred_action = agent.predict(
                        {'observation': obs,
                         'goal_observation': goal,
                         'goal_task_name': goal_task_name}, 
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
                         'goal_observation': goal,
                         'goal_task_name': goal_task_name}
                    )
                    
                    obs, reward, done, info = self.env.step(pred_action.reshape(-1).detach().cpu().numpy())
                    total_reward += reward

        log.info(f"Total reward: {total_reward}")
        
        self.env.close()
        return_dict = self.compute_performance(rewards, results, log_wandb, 'sequential')
        return return_dict
    
    def compute_performance(self, rewards, results, log_wandb: bool = True, eval_type='sequential'):
        """
        Computes the performance metrics based on the rewards and results obtained from testing.

        Args:
            rewards (List[float]): A list of rewards obtained during testing.
            results (List[float]): A list of results obtained during testing.
            log_wandb (bool): Optional. Indicates whether to log the results using Weights & Biases. Defaults to True.
            eval_type (str): Optional. The type of evaluation. Defaults to 'sequential'.

        Returns:
            dict: A dictionary containing the computed performance metrics, including average reward and result.

        """
        avrg_reward = sum(rewards)/len(rewards)
        std_reward = np.array(rewards).std()
        avrg_result = sum(results)/len(results)
        std_result = np.array(results).std()
        
        print('average reward {}'.format(avrg_reward))
        print('average result {}'.format(avrg_result))
        print({"Cond_success_ratio": avrg_result/(avrg_reward + 1e-6)})
        log.info(f"Average reward: {avrg_reward} std: {std_reward}")
        log.info(f"Average result: {avrg_result} std: {std_result}")
        log.info('... finished trained model evaluation of the blockpush environment environment.')
        if log_wandb:
            wandb.log({f"{eval_type}/Average_reward": avrg_reward})
            wandb.log({f"{eval_type}/Average_result": avrg_result})
            wandb.log({f"{eval_type}/Cond_success_ratio": avrg_result/(avrg_reward + 1e-6)})
        
        self.success_rate_1 = self.success_rate_1 / self.eval_n_times
        self.success_rate_2 = self.success_rate_2 / self.eval_n_times
        self.success_rate_3 = self.success_rate_3 / self.eval_n_times
        self.success_rate_4 = self.success_rate_4 / self.eval_n_times
        self.success_rate_5 = self.success_rate_5 / self.eval_n_times
        if log_wandb:
            log.info(f"Success rate 1: {self.success_rate_1}")
            log.info(f"Success rate 2: {self.success_rate_2}")
            log.info(f"Success rate 3: {self.success_rate_3}")
            log.info(f"Success rate 4: {self.success_rate_4}")
            log.info(f"Success rate 5: {self.success_rate_5}")
        else:
            print("Success rate 1: ", self.success_rate_1)
            print("Success rate 2: ", self.success_rate_2)
            print("Success rate 3: ", self.success_rate_3)
            print("Success rate 4: ", self.success_rate_4)
            print("Success rate 5: ", self.success_rate_5)

        if log_wandb:
            log.info(f"---------------------------------------")
        else:
            print("---------------------------------------")
        self.traj_count_sorted = dict(sorted(self.traj_count.items(), key=lambda x: x[1]))
        for traj in self.traj_count_sorted.keys():
            print(f"{traj} : {self.traj_count_sorted[traj]}")
            if log_wandb:
                log.info(f"{traj} : {self.traj_count_sorted[traj]}")


        return_dict = {
            'avrg_reward': avrg_reward,
            'std_reward': std_reward,
            'avrg_result': avrg_result,
            'std_result': std_result
        }
        # return the average reward 
        if log_wandb:
            log.info(f"---------------------------------------")
        else:
            print("---------------------------------------")
        for key in self.solved_tasks:
            print(f"{key} solved {self.solved_tasks[key]} times expected {self.expected_tasks[key]} times")
            # print(f"{key} expected {self.expected_tasks[key]} times")
        return return_dict
    
    def _setup_starting_state(self):
        """
        Sets up the starting state by loading initial joint positions and velocities from files.
        """
        self.init_qpos = np.load(
            Path(self.data_path) / "all_init_qpos.npy"
        )
        self.init_qvel = np.load(
            Path(self.data_path) / "all_init_qvel.npy"
        )
    
    def _start_from_known(self):
        """
        Starts the environment from a known demonstration by randomly selecting an index,
        setting the state based on the corresponding initial joint positions and velocities,
        and returning the observation.

        Returns:
            Observation: The initial observation after setting the state.
        """
        ind = np.random.randint(len(self.init_qpos))
        print(f"Starting from demo {ind}")
        qpos, qvel = self.init_qpos[ind], self.init_qvel[ind]
        self.env.set_state(qpos, qvel)
        obs, _, _, _ = self.env.step(np.zeros(self.cfg.env.action_dim))
        return obs
    
    def _report_result_upon_completion(self, goal_idx=None):
        """
        Reports the results upon completion of the tasks, including completed and incomplete tasks,
        expected tasks based on the goal index, and success rates for different completion indices.

        Args:
            goal_idx (int): Optional. The index of the goal for expected tasks. Defaults to None.
        Returns:
            int: The number of conditional done tasks.
        """
        print("Complete tasks ", self.env.all_completions)
        print("Incomplete tasks ", set(self.env.tasks_to_complete))
        if goal_idx is not None:
            train_idx, val_idx = get_split_idx(
                len(self.relay_traj),
                seed=self.seed,
                train_fraction=self.train_fraction,
            )
            _, _, _, onehot_labels = self.relay_traj[train_idx[goal_idx]]  # T x 7
            expected_mask = onehot_labels.max(0).values.bool().numpy()
            tasks = np.array(self.env.ALL_TASKS)
            expected_tasks = tasks[expected_mask].tolist()
            print("Expected tasks ", expected_tasks)
            conditional_done = set(self.env.all_completions).intersection(
                expected_tasks
            )
            for idx in range (len(self.env.all_completions)):
                if idx == 0:
                    self.success_rate_1 += 1 
                if idx == 1:
                    self.success_rate_2 += 1
                if idx == 2:
                    self.success_rate_3 += 1
                if idx == 3:
                    self.success_rate_4 += 1
                if idx == 4:
                    self.success_rate_5 += 1
            task_list = ', '.join(self.env.all_completions)
            if task_list not in self.used_trajectories:
                self.used_trajectories.append(task_list)
                self.traj_count[task_list] = 1
            else:
                self.traj_count[task_list] += 1

            for task in tasks:
                if task in self.env.all_completions:
                    self.solved_tasks[f"n_{task}"] += 1
                
                if task in expected_tasks:
                    self.expected_tasks[f"n_{task}"] += 1
            
            return len(conditional_done)
        else:
            return len(self.env.all_completions)
    
    def rearrange_array(self, a1, a2):
        """
        Rearranges the elements in `a1` based on the order of elements in `a2`.

        Args:
            a1: The array to be rearranged.
            a2: The array specifying the desired order.
        Returns:
            List: The rearranged array.
        """
        sorted_indices = sorted(range(len(a2)), key=lambda k: a2[k])
        return [a1[i] for i in sorted_indices]


    def return_expert_task_completion(self):
        """
        Returns expert task completion statistics and resets the task-related attributes.
        """
        onehot_labels = self.datasets[0].dataset.dataset.onehot_goals
        for traj in onehot_labels:
            expected_mask = traj.max(0).values.bool().numpy()
            order = traj.max(0).indices.numpy()[expected_mask]
            # print(order)
            tasks = self.all_tasks
            expected_tasks = tasks[expected_mask].tolist()
            expected_tasks = self.rearrange_array(expected_tasks, order)
            for task in expected_tasks:
                task_list = ', '.join(expected_tasks)
                
            if task_list not in self.used_trajectories:
                self.used_trajectories.append(task_list)
                self.traj_count[task_list] = 1
            else:
                self.traj_count[task_list] += 1

            for task in tasks:
                if task in expected_tasks:
                    self.solved_tasks[f"n_{task}"] += 1
                
                if task in expected_tasks:
                    self.expected_tasks[f"n_{task}"] += 1
        
        # get task transistions
        self.get_state_transitions()
        
        self.traj_count_sorted = dict(sorted(self.traj_count.items(), key=lambda x: x[1]))
        for traj in self.traj_count_sorted.keys():
            print(f"{traj} : {self.traj_count_sorted[traj]}")
        print("---------------------------------------")
        for key in self.solved_tasks:
            print(f"{key} solved {self.solved_tasks[key]} times expected {self.expected_tasks[key]} times")
        
        print('done')
        self.reset_tasks()
    
    def get_state_transitions(self):
        """
        Computes state transition probabilities based on the completed tasks and builds a task tree.
        """
        transitions_dict = {}
        transitions_list = []
        self.task_tree = {}
        for traj in self.traj_count:
            completed_tasks_single = traj.split(',')
            completed_tasks = [task.strip() for task in completed_tasks_single]
            
            for idx, task in enumerate(completed_tasks):
                if idx == 0:
                    first_task = task
                    if task not in self.task_tree:
                        self.task_tree[task] = {}
                        self.task_tree[completed_tasks[idx]]['count'] = self.traj_count[traj]
                    else:
                        # self.task_tree[completed_tasks[idx]][completed_tasks[idx + 1]] += 1
                        self.task_tree[completed_tasks[idx]]['count'] += self.traj_count[traj]
                elif idx == 1:
                    second_task = task
                    if task not in self.task_tree[first_task]:
                        self.task_tree[first_task][completed_tasks[idx]] = {}
                        self.task_tree[first_task][completed_tasks[idx]]['count'] = self.traj_count[traj]
                    else:
                        # self.task_tree[first_task][completed_tasks[idx]] += 1
                        self.task_tree[first_task][completed_tasks[idx]]['count'] += self.traj_count[traj]
                elif idx == 2:
                    third_task = task
                    if task not in self.task_tree[first_task][second_task]:
                        self.task_tree[first_task][second_task][completed_tasks[idx]] = {}
                        self.task_tree[first_task][second_task][completed_tasks[idx]]['count'] = self.traj_count[traj]
                    else:
                        # self.task_tree[first_task][second_task][completed_tasks[idx]] += 1
                        self.task_tree[first_task][second_task][completed_tasks[idx]]['count'] += self.traj_count[traj]
                elif idx == len(completed_tasks) - 1:
                    if task not in self.task_tree[first_task][second_task][third_task]:
                        self.task_tree[first_task][second_task][third_task][completed_tasks[idx]] = {}
                        self.task_tree[first_task][second_task][third_task][completed_tasks[idx]]['count'] = self.traj_count[traj]
                    else:
                        # self.task_tree[first_task][second_task][third_task][completed_tasks[idx]] += 1
                        self.task_tree[first_task][second_task][third_task][completed_tasks[idx]]['count'] += self.traj_count[traj]
                
                if idx < len(completed_tasks) - 1:
                    transition = completed_tasks[idx] + ' ' + completed_tasks[idx + 1]
                    if transition not in transitions_dict:
                        transitions_dict[transition] = self.traj_count[traj]
                    else:
                        transitions_dict[transition] += self.traj_count[traj]
                    transitions_list.append(transition)
        # now get all probablities of the state transitions
        for task in self.task_tree:
            self.task_tree[task]['prob'] = self.task_tree[task]['count'] / np.sum(list(self.traj_count.values()))
            if task == 'count' or task == 'prob':
                continue
            for task2 in self.task_tree[task]:
                if task2 == 'count' or task2 == 'prob':
                    continue
                self.task_tree[task][task2]['prob'] = self.task_tree[task][task2]['count'] / self.task_tree[task]['count']
                for task3 in self.task_tree[task][task2]:
                    if task3 == 'count' or task3 == 'prob':
                        continue
                    self.task_tree[task][task2][task3]['prob'] = self.task_tree[task][task2][task3]['count'] / self.task_tree[task][task2]['count']
                    for task4 in self.task_tree[task][task2][task3]:
                        if task4 == 'count' or task4 == 'prob':
                            continue
                        self.task_tree[task][task2][task3][task4]['prob'] = self.task_tree[task][task2][task3][task4]['count'] / self.task_tree[task][task2][task3]['count']
            
        self.tranistions_dict = transitions_dict
        self.transitions_list = transitions_list
        print('done getting transitions')