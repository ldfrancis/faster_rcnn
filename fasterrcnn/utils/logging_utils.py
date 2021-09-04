from abc import abstractmethod
from typing import Any, Dict

import tensorflow as tf
from absl import logging as logger
from fasterrcnn.utils.path_utils import resolve_path


class BaseLogger:
    """The base logger class that handles logging"""

    def __init__(self):
        self.__logger = logger

    @abstractmethod
    def log(self, info_dict):
        raise NotImplementedError

    def __getattr__(self, name):
        return getattr(self.__logger, name)


class TensorboardLogger(BaseLogger):
    """Handles logging of training information to tensorboard and can also replicate
    the same in weight and biases
    """

    def __init__(self, config: Dict[str, Any]):
        """Creates a TensorboardLogger instance

        Args:
            config (Dict[str, Any]): The config dict used to setup faster rcnn
        """
        super(TensorboardLogger, self).__init__()
        self.cfg = config
        tensorboard_logdir = resolve_path(self.cfg.get("log_dir", "~/fasterrcnn/logs"))
        self.global_step = 0

        if self.cfg["log"] and self.cfg["train_type"] == "4step":

            self.step_1_summary_writer = tf.summary.create_file_writer(
                str((tensorboard_logdir / "step_1_rpn").absolute())
            )
            self.step_2_summary_writer = tf.summary.create_file_writer(
                str((tensorboard_logdir / "step_2_detector").absolute())
            )
            self.step_3_summary_writer = tf.summary.create_file_writer(
                str((tensorboard_logdir / "step_3_rpn").absolute())
            )
            self.step_4_summary_writer = tf.summary.create_file_writer(
                str((tensorboard_logdir / "step_4_detector").absolute())
            )
            self.step_to_summary_writer = {
                1: self.step_1_summary_writer,
                2: self.step_2_summary_writer,
                3: self.step_3_summary_writer,
                4: self.step_4_summary_writer,
            }
        elif self.cfg["log"] and self.cfg["train_type"] == "approximate":
            self.approximate_summary_writer = tf.summary.create_file_writer(
                str((tensorboard_logdir / "approximate").absolute())
            )
            self.step_to_summary_writer = {
                1: self.approximate_summary_writer,
            }

        # log to weight and biases
        if self.cfg["log"] and self.cfg.get("wandb"):
            self.to_wandb()

    def log(self, info_dict: Dict[str, Any]) -> None:
        """Logs training info to tensorboard

        Args:
            info_dict (Dict[str, Any]): Dictionary containing info to log to tensorboard
        """

        self.global_step += 1
        global_step = self.global_step
        step = info_dict["trainer_step"]

        message = ""
        for k, v in info_dict.items():
            message += f"{k}:{v:.4f}\t"

        self.warning(message)

        if not self.cfg["log"]:
            return
        else:
            summary_writer = self.step_to_summary_writer[step]

            with summary_writer.as_default():
                for k, v in info_dict.itmes():
                    tf.summary.scalar(k, v, step=global_step)

    def to_wandb(self):
        """Syncs tensorboard logs to weight and biases"""
        try:
            import wandb
        except ModuleNotFoundError as e:
            self.warning(
                (
                    "Failed to sync wandb with tensorboard. Consider installing wandb.\npip install wandb"
                )
            )

        project = self.cfg["wandb"]["project"]
        entity = self.cfg["wandb"]["entity"]

        wandb.init(
            project=project, entity=entity, sync_tensorboard=True, config=self.cfg
        )

        wandb.run.name = (
            f"{self.cfg['experiment_name']}_{self.cfg['rpn']['backbone']}_"
            f"{self.cfg['dataset']}"
        )
