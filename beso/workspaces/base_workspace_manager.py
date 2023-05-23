import logging
import abc
import os
import copy

import torch 
import numpy as np
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import wandb

from beso.agents.diffusion_agents.beso_agent import BesoAgent
from beso.agents.diffusion_agents.k_diffusion.classifier_free_sampler import ClassifierFreeSampleModel


log = logging.getLogger(__name__)


class BaseWorkspaceManger(abc.ABC):

    def __int__(
            self,
            seed: int,
            device: str
    ):
        self.seed = seed
        self.device = device
        self.working_dir = os.getcwd()
        self.env_name = 'BaseEnvironment'

    @abc.abstractmethod
    def test_agent(self, agent):
        pass

    @staticmethod
    def split_datasets(dataset, train_fraction=0.9, random_seed=42):
        dataset_length = len(dataset)
        lengths = [
            int(train_fraction * dataset_length),
            dataset_length - int(train_fraction * dataset_length),
        ]
        train_set, val_set = random_split(
            dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
        )
        return train_set, val_set

    def compare_sampler_types(
        self, 
        agent, 
        num_runs, 
        num_steps_per_run, 
        log_wandb: bool = True, 
        n_inference_steps=None,
        get_mean=None,
        store_path=None
    ):
        
        # store the old values
        old_num_rums = self.eval_n_times
        old_n_steps  = self.eval_n_steps
        # overwrite these variables for the method
        self.eval_n_times = num_runs
        self.eval_n_steps = num_steps_per_run
        
        if not isinstance(agent, BesoAgent):
            raise ValueError('This method requires BesoAgent type!')
        samplers = ['euler', 'ancestral', 'euler_ancestral', 'heun', 'lms', 'dpm', 
                    'dpmpp_2s_ancestral', 'dpmpp_2m']

        avrg_rewards = []
        results = []
        std_rewards = []
        std_results = []

        for sampler_type in samplers:
            return_dict = self.test_agent(
                agent, 
                log_wandb=log_wandb, 
                new_sampler_type=sampler_type,
                get_mean=get_mean,
                n_inference_steps=n_inference_steps
            )
            avrg_rewards.append(round(return_dict['avrg_reward'], 2))
            results.append(round(return_dict['avrg_result'], 2))
            std_rewards.append(round(return_dict['std_reward'], 2))
            std_results.append(round(return_dict['std_result'], 2))

        for idx, sampler_type in enumerate(samplers):
            log.info(sampler_type + f' reward: {avrg_rewards[idx]} std: {std_rewards[idx]},\
                     result {results[idx]}, std: {std_results[idx]}')
            
        print("done comparing all sampler types!")
        self.eval_n_times = old_num_rums
        self.eval_n_steps = old_n_steps
    
        if store_path is not None:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=400)


            samplers = ['Euler', 'AC', 'EA', 'Heun', 'LMS', 'DPM', 'DPM++\n(2S) A', 'DPM++\n(2M)']
                
            labels = samplers
            x = np.arange(len(labels)) # the label locations

            width = 0.25 # the width of the bars

            rects1 = ax.bar(x - width/2, avrg_rewards, width, yerr = std_rewards, ecolor='black', alpha=0.5, label='Reward')
            rects2 = ax.bar(x + width/2, results, width, yerr = std_results, ecolor='black', label='Result')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(f'Average performance of {num_runs} runs')
            ax.set_xlabel(f'Sampler type')

            ax.set_title(f'Sampler comparisons for {n_inference_steps} denoising steps')

            ax.set_xticks(x, labels)

            # rotate 45 degrees

            # for tick in ax.get_xticklabels():

            # tick.set_rotation(45)

            # ax.legend(loc='upper left')

            ax.bar_label(rects1, padding=2)
            ax.bar_label(rects2, padding=2)
            ax.yaxis.grid(True)
            # fig.tight_layout()
            plt.ylim([0, round(max(max(np.array(std_results) + np.array(results)),
            max(np.array(std_rewards) + np.array(avrg_rewards))) + 0.3, 1 )])
            plot_name = 'Sampler_comparison_' + str(n_inference_steps) + 'diff_steps.png'
            plot_store_name = os.path.join(store_path, plot_name)
            plt.savefig(plot_store_name)
            plt.close()
    
    def compare_noisy_sampler(
        self, 
        agent, 
        num_runs, 
        num_steps_per_run, 
        log_wandb: bool = True, 
        n_inference_steps=None,
        get_mean=None,
        store_path=None
    ):
        
        # store the old values
        old_num_rums = self.eval_n_times
        old_n_steps  = self.eval_n_steps
        # overwrite these variables for the method
        self.eval_n_times = num_runs
        self.eval_n_steps = num_steps_per_run
        
        if not isinstance(agent, BesoAgent):
            raise ValueError('This method requires BesoAgent type!')
        samplers = ['euler', 'dpm', 'dpmpp_2m', 'euler_ancestral', 'ancestral', 'dpmpp_2m_sde']

        avrg_rewards = []
        results = []
        std_rewards = []
        std_results = []

        for sampler_type in samplers:
            return_dict = self.test_agent(
                agent, 
                log_wandb=log_wandb, 
                new_sampler_type=sampler_type,
                get_mean=get_mean,
                n_inference_steps=n_inference_steps
            )
            avrg_rewards.append(round(return_dict['avrg_reward'], 2))
            results.append(round(return_dict['avrg_result'], 2))
            std_rewards.append(round(return_dict['std_reward'], 2))
            std_results.append(round(return_dict['std_result'], 2))

        for idx, sampler_type in enumerate(samplers):
            log.info(sampler_type + f' reward: {avrg_rewards[idx]} std: {std_rewards[idx]},\
                     result {results[idx]}, std: {std_results[idx]}')
            
        print("done comparing all sampler types!")
        self.eval_n_times = old_num_rums
        self.eval_n_steps = old_n_steps
        
        
        deterministc_reward = avrg_rewards[0:3]
        noisy_reward = avrg_rewards[3:]
        deterministc_result = results[:3]
        noisy_result = results[3:]
        
        std_deterministc_reward = std_rewards[0:3]
        std_noisy_reward = std_rewards[3:]
        std_result_deterministc = std_results[:3]
        std_result_noisy = std_results[3:]
        
        from tueplots import bundles
        from tueplots import bundles, axes
        plt.rcParams.update(bundles.icml2022())
        plt.rcParams.update({"figure.dpi": 300})
        
        if store_path is not None:
            with plt.rc_context({**bundles.icml2022(family="sans-serif", column="full", nrows=1), **axes.lines()}):
                fig, axs = plt.subplots(nrows=1, ncols=2)
                
                for i in range(2):
                    ax = axs[i]
                    # samplers = ['Euler', 'AC', 'EA', 'Heun', 'LMS', 'DPM', 'DPM++\n(2S) A', 'DPM++\n(2M)']
                    # samplers = ['Euler', 'Euler\nAC', 'DPM', 'DPM\nAC', 'DPM++\n(2M)', 'DPM++\n(2M)AC']  
                    samplers = ['Euler','DPM', 'DPM++\n(2M)']  
                    labels = samplers
                    x = np.arange(len(labels)) # the label locations

                    width = 0.25 # the width of the bars

                    if i == 0:
                        rects1 = ax.bar(x - width/2, deterministc_reward, width, yerr = std_deterministc_reward, ecolor='black', alpha=0.5, label='Deterministic Sampler')
                        rects2 = ax.bar(x + width/2, noisy_reward, width, yerr = std_noisy_reward, ecolor='black', label='with Noise')
                    else:
                        rects1 = ax.bar(x - width/2, deterministc_result, width, yerr = std_result_deterministc, ecolor='black', alpha=0.5, label='Deterministic Sampler')
                        rects2 = ax.bar(x + width/2, noisy_result, width, yerr = std_result_noisy, ecolor='black', label='with Noise')
                    # Add some text for labels, title and custom x-axis tick labels, etc.
                    if ax == 0:
                        ax.set_ylabel(f'Average reward of {num_runs} runs', fontsize=16)
                    else:
                        ax.set_ylabel(f'Average result of {num_runs} runs', fontsize=16)
                    # ax.set_xlabel(f'Sampler type')

                    # ax.set_title(f'Sampler comparisons for {n_inference_steps} denoising steps')

                    ax.set_xticks(x, labels, fontsize=16)

                    # rotate 45 degrees

                    # for tick in ax.get_xticklabels():

                    # tick.set_rotation(45)
                    plt.grid(b=True, which='major', color='0.2', linestyle='-')
                    plt.grid(b=True, which='minor', color='0.2', linestyle='-')
                    ax.patch.set_edgecolor('black')
                    ax.patch.set_linewidth('1')
                    plt.yticks(fontsize=16)
                    plt.xticks(fontsize=16)
                    plt.rc('ytick', labelsize=16)
                    plt.rc('xtick', labelsize=16)

                    # ax.bar_label(rects1, padding=2)
                    # ax.bar_label(rects2, padding=2)
                    ax.yaxis.grid(True)
                    
                lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
                lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
                plt.legend(lines, labels, loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
                    bbox_transform = plt.gcf().transFigure, fontsize=16, ncol=4, frameon=False)
                # fig.legend(loc=(0, 1.02), fontsize=16, ncol=1, frameon=False)
                # plt.figlegend(loc=(0, 1.02), fontsize=16, ncol=1)
                # fig.tight_layout()
                plt.tight_layout()
                plt.ylim([0, round(max(max(np.array(std_results) + np.array(results)),
                    max(np.array(std_rewards) + np.array(avrg_rewards))) + 0.3, 1 )])
                plot_name = 'Noise_Sampler_comparison_' + str(n_inference_steps) + f'diff_steps_of_{num_runs}_runs.png'
                plot_store_name = os.path.join(store_path, plot_name)
                plt.savefig(plot_store_name)
                plot_name = 'Noise_Sampler_comparison_' + str(n_inference_steps) + f'diff_steps_of_{num_runs}_runs.pdf'
                plot_store_name = os.path.join(store_path, plot_name)
                plt.savefig(plot_store_name)
                plt.close()
    
    def compare_sde_sampling(
        self, 
        agent, 
        num_runs, 
        sampler_type,
        num_steps_per_run, 
        churn_list: list,
        n_inference_steps= None,
        log_wandb: bool = True, 
        get_mean=None,
        store_path=None
        ):
        # store the old values
        old_num_rums = self.eval_n_times
        old_n_steps  = self.eval_n_steps
        # overwrite these variables for the method
        self.eval_n_times = num_runs
        self.eval_n_steps = num_steps_per_run
        
        if not isinstance(agent, BesoAgent):
            raise ValueError('This method requires BesoAgent type!')
        
        avrg_rewards = []
        std_rewards = []
        results = []
        std_results = []
        
        # update the model class of the agent
        default_model = copy.deepcopy(agent.model)
        for idx, churn in enumerate(churn_list):
            return_dict = self.test_agent(
                agent, 
                log_wandb=log_wandb, 
                new_sampler_type=sampler_type,
                get_mean=get_mean,
                n_inference_steps=n_inference_steps,
                extra_args={'s_churn': churn}
            )
            avrg_rewards.append(round(return_dict['avrg_reward'], 2))
            results.append(round(return_dict['avrg_result'], 2))
            std_rewards.append(round(return_dict['std_reward'], 2))
            std_results.append(round(return_dict['std_result'], 2))
            
        
        if store_path is not None:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=400)
            labels = []
            for churn_value in churn_list:
                labels.append( f'S_churn factor: {churn_value}') 
                
            x = np.arange(len(labels))  # the label locations
            width = 0.25  # the width of the bars

            rects1 = ax.bar(x - width/2, avrg_rewards, width, yerr = std_rewards, ecolor='black',  alpha=0.5,  label='Reward')
            rects2 = ax.bar(x + width/2, results, width,  yerr = std_results, ecolor='black', label='Result')
            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(f'Average performance of {num_runs} runs')
            ax.set_title(f'SDE_comparison for {n_inference_steps} denoising steps and ' + sampler_type + ' sampling')
            ax.set_xticks(x, labels)
            # ax.legend()

            ax.bar_label(rects1, padding=3)
            ax.bar_label(rects2, padding=3)
            ax.yaxis.grid(True)
            fig.tight_layout()
            plt.ylim([0, round(
                max(max(np.array(std_results) + np.array(results)),
                    max(np.array(std_rewards) + np.array(avrg_rewards))
                    )+ 0.3, 1 
                )])
            
            plot_name = 'S_churn_list_' + f'{len(churn_list)}_lambdas_' + sampler_type + '_'  + str(n_inference_steps) + 'diff_steps.png'
            plot_store_name = os.path.join(store_path, plot_name)
            plt.savefig(plot_store_name)
            plt.close()
    
    def compare_classifier_free_guidance(
        self, 
        agent, 
        num_runs, 
        sampler_type,
        num_steps_per_run, 
        cond_lambda_list: list,
        n_inference_steps= None,
        log_wandb: bool = True, 
        get_mean=None,
        store_path=None,
        extra_args={}
        ):
        # store the old values
        old_num_rums = self.eval_n_times
        old_n_steps  = self.eval_n_steps
        # overwrite these variables for the method
        self.eval_n_times = num_runs
        self.eval_n_steps = num_steps_per_run
        
        if not isinstance(agent, BesoAgent):
            raise ValueError('This method requires BesoAgent type!')
        
        avrg_rewards = []
        std_rewards = []
        results = []
        std_results = []
        
        # update the model class of the agent
        default_model = copy.deepcopy(agent.model)
        for idx, cond_lambda in enumerate(cond_lambda_list):
            agent.model = ClassifierFreeSampleModel(default_model, cond_lambda=cond_lambda)
            return_dict = self.test_agent(
                agent, 
                log_wandb=log_wandb, 
                new_sampler_type=sampler_type,
                get_mean=get_mean,
                n_inference_steps=n_inference_steps,
                extra_args=extra_args
            )
            avrg_rewards.append(round(return_dict['avrg_reward'], 2))
            results.append(round(return_dict['avrg_result'], 2))
            std_rewards.append(round(return_dict['std_reward'], 2))
            std_results.append(round(return_dict['std_result'], 2))
            print(f'Done with cond value of {cond_lambda}')
            log.info(f"Average reward: {return_dict['avrg_reward']} std: {return_dict['std_reward']}")
            log.info(f"Average result: {return_dict['avrg_result']} std: {return_dict['std_result']}")
            if log_wandb:
                wandb.log({"Average_reward": round(return_dict['avrg_reward'], 2)})
                wandb.log({"Average_result": round(return_dict['avrg_result'], 2)})
                wandb.log({"Cond_success_ratio": round(return_dict['avrg_result'], 2)/round(return_dict['avrg_reward'], 2)})
            if log_wandb:
                log.info(f"---------------------------------------")
        
        if store_path is not None:
            
            # now plot the results 
            fig, ax = plt.subplots(figsize=(10, 5), dpi=400)
            labels = []
            for lambda_value in cond_lambda_list:
                labels.append( f'{lambda_value}') 
                
            x = np.arange(len(labels))  # the label locations
            width = 0.25 # the width of the bars

            rects1 = ax.bar(x - width/2, avrg_rewards, width, yerr = std_rewards, ecolor='black',  alpha=0.5,  label='Reward')
            rects2 = ax.bar(x + width/2, results, width,  yerr = std_results, ecolor='black', label='Result')
            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(f'Average performance of {num_runs} runs')
            ax.set_title(f'Comparison for {n_inference_steps} denoising steps and ' + sampler_type + ' sampling')
            ax.set_xticks(x, labels)
            # ax.legend()

            ax.bar_label(rects1, padding=3)
            ax.bar_label(rects2, padding=3)
            ax.yaxis.grid(True)
            fig.tight_layout()
            plt.ylim([0, round(
                max(max(np.array(std_results) + np.array(results)),
                    max(np.array(std_rewards) + np.array(avrg_rewards))
                    )+ 0.1, 1 
                )])
            
            plot_name = 'Classifier_free_comparison_' + f'{len(cond_lambda_list)}_lambdas_' + sampler_type + '_'  + str(n_inference_steps) + 'diff_steps.png'
            plot_store_name = os.path.join(store_path, plot_name)
            plt.savefig(plot_store_name)
            plt.close()
    
    def compare_kde_vs_mean_vs_single(
        self, 
        agent, 
        num_runs, 
        sampler_type,
        num_steps_per_run, 
        n_inference_steps= None,
        log_wandb: bool = True, 
        get_mean=None,
        store_path=None,
        extra_args={}
        ):
        # store the old values
        old_num_rums = self.eval_n_times
        old_n_steps  = self.eval_n_steps
        # overwrite these variables for the method
        self.eval_n_times = num_runs
        self.eval_n_steps = num_steps_per_run
        
        if not isinstance(agent, BesoAgent):
            raise ValueError('This method requires BesoAgent type!')
        
        avrg_rewards = []
        std_rewards = []
        results = []
        std_results = []
        
        sampler_strategies = ['single', 'mean', 'kde']
        # update the model class of the agent
        default_model = copy.deepcopy(agent.model)
        for idx, cond_lambda in enumerate(sampler_strategies):
            
            if sampler_strategies == 'single':
                get_mean = None
                agent.use_kde = False
            elif sampler_strategies == 'mean':
                get_mean = get_mean
                agent.use_kde = False
                
            elif sampler_strategies == 'kde':
                get_mean = get_mean
                agent.use_kde = True
                
            return_dict = self.test_agent(
                agent, 
                log_wandb=log_wandb, 
                new_sampler_type=sampler_type,
                get_mean=get_mean,
                n_inference_steps=n_inference_steps,
                extra_args=extra_args
            )
            avrg_rewards.append(round(return_dict['avrg_reward'], 2))
            results.append(round(return_dict['avrg_result'], 2))
            std_rewards.append(round(return_dict['std_reward'], 2))
            std_results.append(round(return_dict['std_result'], 2))
            
        
        if store_path is not None:
            fig, ax = plt.subplots()
            labels = ['single', f'mean: {get_mean}', f'KDE: {get_mean}']

            
            x = np.arange(len(labels))  # the label locations
            width = 0.35  # the width of the bars

            rects1 = ax.bar(x - width/2, avrg_rewards, width, yerr = std_rewards, ecolor='black',  alpha=0.5,  label='Reward')
            rects2 = ax.bar(x + width/2, results, width,  yerr = std_results, ecolor='black', label='Result')
            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(f'Average performance of {num_runs} runs')
            ax.set_title(f'Comparison for {n_inference_steps} steps and ' + sampler_type + ' sampling')
            ax.set_xticks(x, labels)
            # ax.legend()

            ax.bar_label(rects1, padding=3)
            ax.bar_label(rects2, padding=3)
            ax.yaxis.grid(True)
            fig.tight_layout()
            plt.ylim([0, round(
                max(max(np.array(std_results) + np.array(results)),
                    max(np.array(std_rewards) + np.array(avrg_rewards))
                    )+ 0.1, 1 
                )])
            
            plot_name = 'Classifier_free_comparison_' + 'Generation_strategy_comparison_for_' + sampler_type + '_' + str(n_inference_steps) + '_diff_steps.png'
            plot_store_name = os.path.join(store_path, plot_name)
            plt.savefig(plot_store_name)
            plt.close()

    def compare_sampler_types_over_n_steps(
        self, 
        agent, 
        num_runs, 
        steps_list, 
        log_wandb: bool = True, 
        samplers_list=None,
        get_mean=None,
        store_path=None,
        extra_args={}
    ):  
        if samplers_list is None:
            samplers = ['euler', 'ancestral', 'euler_ancestral', 'heun', 'lms', 'dpm', 
                    'dpmpp_2s_ancestral', 'dpmpp_2m']
        else:
            samplers = samplers_list
        # store the old values
        old_num_rums = self.eval_n_times
        old_n_steps  = self.eval_n_steps
        # overwrite these variables for the method
        self.eval_n_times = num_runs

        result_array = np.zeros((len(samplers), len(steps_list)))
        reward_array = np.zeros((len(samplers), len(steps_list)))
        reward_std_array = np.zeros((len(samplers), len(steps_list)))
        result_std_array = np.zeros((len(samplers), len(steps_list)))

        for idx, sampler_type in enumerate(samplers):
 
            if not isinstance(agent, BesoAgent):
                raise ValueError('This method requires BesoAgent type!')

            for k, n_steps in enumerate(steps_list):
                return_dict = self.test_agent(
                    agent, 
                    log_wandb=log_wandb, 
                    new_sampler_type=sampler_type,
                    get_mean=get_mean,
                    n_inference_steps=n_steps,
                    extra_args=extra_args
                )

                result_array[idx, k] = round(return_dict['avrg_result'], 3)
                reward_array[idx, k] = round(return_dict['avrg_reward'], 3)
                reward_std_array[idx, k] = round(return_dict['std_reward'], 3)
                result_std_array[idx, k] = round(return_dict['std_result'], 3)
                
        print("done comparing all sampler types for all different inference steps!")


        self.eval_n_times = old_num_rums
        self.eval_n_steps = old_n_steps    

        if store_path is not None:
            
            fig, ax = plt.subplots()
            samplers = [ 
                     'Euler', 'AC', 'EA', 'Heun', 'LMS', 'DPM', 'DPM++\n(2S) A', 'DPM++\n(2M)'
                        ]
            labels = samplers
            for idx, sampler in enumerate(samplers):
                ax.plot(steps_list, result_array[idx, :], label=labels[idx])
                # ax.fill_between(steps_list, result_array[idx, :] - result_std_array[idx, :], result_array[idx, :] + result_std_array[idx, :], alpha=0.15)
            ax.set_xlabel('Inference steps')
            ax.set_ylabel('Average reward')
            # ax.set_yscale('log')
            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(f'Average result of {num_runs} runs')
            ax.set_title(f'Sampler comparisons for different denoising steps')
            ax.legend()
            
            plt.ylim([0, 4.5])
            plt.xlim([min(steps_list), max(steps_list)])
                
            ax.yaxis.grid(True)
            fig.tight_layout()
            
            plot_name = 'Sampler_comparison_' + f'_{len(steps_list)}_different_steps.png'
            plot_store_name = os.path.join(store_path, plot_name)
            plt.savefig(plot_store_name)
            plt.close()
        
        if store_path is not None:
            fig, ax = plt.subplots()
            samplers = [ 
                     'Euler', 'AC', 'EA', 'Heun', 'LMS', 'DPM', 'DPM++\n(2S) A', 'DPM++\n(2M)'
                        ]
            labels = samplers
            for idx, sampler in enumerate(samplers):
                ax.plot(steps_list, result_array[idx, :], label=labels[idx])
                # ax.fill_between(steps_list, result_array[idx, :] - result_std_array[idx, :], result_array[idx, :] + result_std_array[idx, :], alpha=0.15)
            ax.set_xlabel('Inference steps')
            ax.set_ylabel('Average reward')
            ax.set_yscale('log')
            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(f'Average result of {num_runs} runs')
            ax.set_title(f'Sampler comparisons for different denoising steps')
            ax.legend()
            
            plt.ylim([0, 4.5])
            plt.xlim([min(steps_list), max(steps_list)])
                
            ax.yaxis.grid(True)
            fig.tight_layout()
            
            plot_name = 'Sampler_comparison_' + f'_{len(steps_list)}_different_steps_2.png'
            plot_store_name = os.path.join(store_path, plot_name)
            plt.savefig(plot_store_name)
            plt.close()
        
        if store_path is not None:
            fig, ax = plt.subplots()
            samplers = [ 
                     'Euler', 'AC', 'EA', 'Heun', 'LMS', 'DPM', 'DPM++\n(2S) A', 'DPM++\n(2M)'
                        ]
            labels = samplers
            for idx, sampler in enumerate(samplers):
                ax.plot(steps_list, result_array[idx, :], label=labels[idx])
                ax.fill_between(steps_list, result_array[idx, :] - result_std_array[idx, :], result_array[idx, :] + result_std_array[idx, :], alpha=0.15)
            ax.set_xlabel('Inference steps')
            ax.set_ylabel('Average reward')
            ax.set_yscale('log')
            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(f'Average result of {num_runs} runs')
            ax.set_title(f'Sampler comparisons for different denoising steps')
            ax.legend()
            
            plt.ylim([0, 4.5])
            plt.xlim([min(steps_list), max(steps_list)])
                
            ax.yaxis.grid(True)
            fig.tight_layout()
            
            plot_name = 'Sampler_comparison_' + f'_{len(steps_list)}_different_steps_2.png'
            plot_store_name = os.path.join(store_path, plot_name)
            plt.savefig(plot_store_name)
            plt.close()
        
            # save the np array
            np.save(os.path.join(store_path, 'result_array'), result_array)
            np.save(os.path.join(store_path, 'reward_array'), reward_array)
            np.save(os.path.join(store_path, 'result_std_array'), result_std_array)
            np.save(os.path.join(store_path, 'reward_std_array'), reward_std_array)