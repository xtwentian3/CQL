import abc
from collections import OrderedDict

import gtimer as gt

from rlkit.core import logger, eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector
import numpy as np
import torch


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            log_add,
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []
        self.evaluations = []
        self.log_add = log_add

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self, epoch):
        # if not self.trainer.discrete:
        #     snapshot = self._get_snapshot()
        #     logger.save_itr_params(epoch, snapshot)
        #     gt.stamp('saving')
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def save_model(self, name):
        self._save_model(f"{self.log_add}/{name}")

    def _save_model(self, filename):
        torch.save(self.trainer.policy.state_dict(), filename + "_policy")
        torch.save(self.trainer.policy_optimizer.state_dict(), filename + "_policy_optimizer")

        torch.save(self.trainer.qf1.state_dict(), filename + "_qf1")
        torch.save(self.trainer.qf1_optimizer.state_dict(), filename + "_qf1_optimizer")

        torch.save(self.trainer.qf2.state_dict(), filename + "_qf2")
        torch.save(self.trainer.qf2_optimizer.state_dict(), filename + "_qf2_optimizer")

    def _load_model(self, filename):
        self.trainer.policy.load_state_dict(torch.load(filename + "_policy"))
        self.trainer.policy_optimizer.load_state_dict(torch.load(filename + "_policy_optimizer"))

        self.trainer.qf1.load_state_dict(torch.load(filename + "_qf1"))
        self.trainer.qf1_optimizer.load_state_dict(torch.load(filename + "_qf1_optimizer"))
        self.trainer.target_qf1 = copy.deepcopy(self.trainer.qf1)

        self.trainer.qf2.load_state_dict(torch.load(filename + "_qf2"))
        self.trainer.qf2_optimizer.load_state_dict(torch.load(filename + "_qf2_optimizer"))
        self.trainer.target_qf2 = copy.deepcopy(self.trainer.qf2)

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)
        """
        Evaluation
        """
        eval_paths = self.eval_data_collector.get_epoch_paths()
        sta = eval_util.get_generic_path_information(eval_paths)

        self.evaluations = np.append(self.evaluations, sta['Average Returns'])
        np.save(f"{self.log_add}/reward_tran", self.evaluations)
        print(sta['Average Returns'])

        # 阶段性保存模型
        if epoch>0 and epoch % 1000 == 0:
            self.save_model(f"num_{epoch}")

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
