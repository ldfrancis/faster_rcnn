from fasterrcnn.frcnn import FRCNN
from fasterrcnn.trainer import Trainer


class TestTrainer:
    def test_trainer(self, dataset, cfg):
        frcnn = FRCNN(cfg)
        train_dataset, val_dataset, _ = dataset, dataset, dataset

        cfg["trainer"]["log"] = False

        cfg["trainer"]["train_type"] = "approximate"
        trainer = Trainer(frcnn, cfg["trainer"])

        assert (
            train_dataset is not None
            and val_dataset is not None
            and trainer is not None
        )

        # approximate training
        trainer.cfg["train_type"] = "approximate"
        trainer.train(train_dataset)

        # 4 step training
        trainer.cfg["train_type"] = "4step"
        trainer.train(train_dataset)
