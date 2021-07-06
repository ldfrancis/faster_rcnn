import tensorflow as tf
from tensorflow import Variable
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer


def create_checkpoint(
    checkpoint_path,
    max_to_keep,
    rpn: Model,
    detector: Model,
    backbone_head: Model,
    backbone_tail: Model,
    rpn_optimizer: Optimizer,
    detector_optimizer: Optimizer,
    detector_backbone_optimizer: Optimizer,
    backbone_head_optimizer: Optimizer,
    backbone_tail_optimizer: Optimizer,
    best_score: Variable,
    trainer_step: Variable,
    metric: str = "map",
    train_type: str = "approximate",
):
    ckpt = tf.train.Checkpoint(
        rpn=rpn,
        detector=detector,
        backbone_head=backbone_head,
        backbone_tail=backbone_tail,
        detector_optimizer=detector_optimizer,
        rpn_optimizer=rpn_optimizer,
        detector_backbone_optimizer=detector_backbone_optimizer,
        backbone_head_optimizer=backbone_head_optimizer,
        backbone_tail_optimizer=backbone_tail_optimizer,
        best_score=best_score,
        metric=tf.Variable(metric),
        train_type=tf.Variable(train_type),
        trainer_step=trainer_step,
    )
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=max_to_keep
    )

    return Checkpoint(ckpt, ckpt_manager, metric=metric)


class Checkpoint:
    def __init__(self, checkpoint, checkpoint_manager, metric="map"):
        self._ckpt = checkpoint
        self._ckpt_manager = checkpoint_manager
        self._metric = metric
        self.best_score = self._ckpt.best_score
        self.trainer_step = self._ckpt.trainer_step
        self.save_path = None

    def update(self, score) -> bool:
        _updated = False
        if self.should_update(score):
            self.best_score.assign(score)
            self.save_path = self._ckpt_manager.save()
            _updated = True
        return _updated

    def should_update(self, score) -> bool:
        _should_update = False
        if self.metric == "map":
            _should_update = tf.less(self.best_score, score)
        else:
            _should_update = tf.greater(self.best_score, score)

        return _should_update

    def restore(self, chkpt_path):
        self._ckpt.restore(chkpt_path)

    @property
    def train_type(self):
        return self._ckpt.train_type.numpy()

    @property
    def metric(self):
        return self._ckpt.metric.numpy()
