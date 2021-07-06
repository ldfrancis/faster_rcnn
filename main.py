import logging
from pathlib import Path
from typing import Any, Dict

import click
import tensorflow as tf
import yaml

from fasterrcnn.frcnn import FRCNN
from fasterrcnn.trainer import Trainer
from fasterrcnn.utils.config_utils import load_config
from fasterrcnn.utils.data_utils import obtain_dataset

logging.basicConfig(level=logging.DEBUG)


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
    detector_cfg = cfg["detector"]
    rpn_cfg = cfg["rpn"]
    frcnn = FRCNN({"detector": detector_cfg, "rpn": rpn_cfg})
    cfg["trainer"]["log"] = log
    cfg["trainer"]["dataset"] = dataset

    if train:
        trainer = Trainer(frcnn, cfg["trainer"])
        (train_dataset, val_dataset, _), _ = obtain_dataset(dataset)
        trainer.train(train_dataset, val_dataset)
    else:
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

            image = tf.io.read_file(f"{inp_path.absolute()}")
            image = tf.io.decode_image(image)

            boxes, scores = frcnn(image)

            logging.info(boxes, scores)
