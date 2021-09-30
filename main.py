import logging
from pathlib import Path
from typing import Any, Dict

import click
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from absl import logging as logger

from fasterrcnn.frcnn import FRCNN
from fasterrcnn.trainer import Trainer
from fasterrcnn.utils.checkpoint_utils import restore_fasterrcnn
from fasterrcnn.utils.config_utils import load_config
from fasterrcnn.utils.data_utils import obtain_dataset
from fasterrcnn.utils.data_utils.data_utils import display_image, obtain_class_names
from fasterrcnn.utils.data_utils.tfds_utils import modify_image_size


@click.command()
@click.option(
    "--train",
    "-t",
    show_default=True,
    is_flag=True,
    help="Decide whether to train  model.",
)
@click.option(
    "--log",
    "-l",
    show_default=True,
    is_flag=True,
    help="Decide whether to log to tensorboard.",
)
@click.option(
    "--dataset",
    "-d",
    default="voc",
    show_default=True,
    type=click.Choice(["voc", "coco"]),
    help="Dataset to use.",
)
@click.option(
    "--input",
    "-i",
    show_default=True,
    type=str,
    help="input image path or path to folder containing images",
)
@click.option(
    "--output",
    "-o",
    show_default=True,
    type=str,
    help="path to folder where output results would be saved to",
)
def main(train, log, dataset, input, output):

    cfg = load_config("./config.yaml")
    frcnn = FRCNN(cfg)
    cfg["trainer"]["log"] = log
    cfg["trainer"]["dataset"] = dataset

    if train:
        trainer = Trainer(frcnn, cfg["trainer"])
        (train_dataset, val_dataset, _), _ = obtain_dataset(dataset)
        trainer.train(train_dataset, val_dataset)
    else:
        restore_fasterrcnn(frcnn)
        output = output or "./output"
        assert input is not None and output is not None, (
            "Please provide an input and output paths for prediction.\nExample:\n "
            "fasterrcnn -i img.jpg -o output-folder"
        )
        input_path = Path(input)
        output_path = Path(output)
        inputs = [input_path] if not input_path.is_dir() else input_path.iterdir()
        output_path.mkdir(parents=True, exist_ok=True)

        for inp_path in inputs:
            logger.warn(f"Detecting objects in {inp_path.absolute()}")
            image, boxes, scores = frcnn(f"{inp_path.absolute()}")

            _ = display_image(image, boxes, obtain_class_names(cfg["dataset"]), scores)
            plt.savefig(output_path / inp_path.name)
