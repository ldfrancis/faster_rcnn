from pathlib import Path
from typing import Any, Dict

import click
import tensorflow as tf
import yaml

from faster_rcnn.frcnn import FRCNN
from faster_rcnn.trainer import Trainer
from faster_rcnn.utils.data_utils import obtain_dataset


def load_config(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        loaded_yaml = yaml.load(file, Loader=yaml.FullLoader)
    return loaded_yaml


@click.command()
@click.option(
    "--train",
    "-tr",
    show_default=True,
    is_flag=True,
    help="Decide whether to train  model.",
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
    default="./input/img.jpg",
    show_default=True,
    type=str,
    help="input image path or path to folder containing images",
)
@click.option(
    "--output",
    "-o",
    default="./output",
    show_default=True,
    type=str,
    help="path to folder where output results would be saved to",
)
def main(train, dataset, input, output):
    cfg = load_config("./config.yaml")
    detector_cfg = cfg["detector"]
    rpn_cfg = cfg["rpn"]
    frcnn = FRCNN({"detector": detector_cfg, "rpn": rpn_cfg})
    if train:
        trainer = Trainer(frcnn)
        (train_dataset, val_dataset, _), _ = obtain_dataset(dataset)
        trainer.train(train_dataset, val_dataset)
    else:
        input_path = Path(input)
        output_path = Path(output)
        inputs = [input_path] if not input_path.is_dir() else input_path.iterdir()
        output_path.mkdir(parents=True, exist_ok=True)

        for inp_path in inputs:
            image = tf.io.read_file(inp_path)
            image = tf.io.decode_image(image)

            boxes, scores = frcnn(image)

            print(boxes, scores)
