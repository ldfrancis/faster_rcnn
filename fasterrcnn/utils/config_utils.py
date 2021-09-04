from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict

import wandb
import yaml


def load_yaml(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        loaded_yaml = yaml.load(file, Loader=yaml.FullLoader)
    return loaded_yaml


def get_captionwiz_dir() -> Path:
    captionwiz_dir = Path.home() / "captionwiz"
    return captionwiz_dir


def get_datetime(format="%Y%m%d%H%M%S") -> str:
    now = datetime.now()
    dtime = now.strftime(format)
    return dtime


def setup_wandb(cfg):
    if cfg["wandb"]["use"]:
        project = cfg["wandb"]["project"]
        entity = cfg["wandb"]["entity"]
        wandb.init(project=project, entity=entity, sync_tensorboard=True, config=cfg)
        wandb.run.name = f"{cfg['name']}_{cfg['caption_model']}_{cfg['dataset']}"


def load_config(file_path: Path) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as file:
        loaded_yaml = yaml.load(file, Loader=yaml.FullLoader)
    return loaded_yaml


frcnn_default_config = {
    "rpn": {
        "name": "rpn",
        "backbone": "resnet101",
        "anchor_base_size": 256,
        "anchor_ratios": [0.5, 1, 2],
        "anchor_scales": [0.125, 0.25, 0.5, 1, 2],
        "base_conv_channels": 512,
        "input_channels": 1024,
        "score_thresh": 0.7,
        "nms_threshold": 0.7,
        "top_n": 2000,
        "pool_size": 7,
        "stride": 16,
    },
    "detector": {
        "name": "detector",
        "num_classes": 20,
        "input_channels": 2048,
        "top_n": 19,
        "score_thresh": 0.7,
        "nms_threshold": 0.7,
    },
    "image_base_size": 600,
    "dataset": "voc",
}

trainer_default_config: Dict[str, Any] = {
    **frcnn_default_config,
    "trainer": {
        "name": "trainer",
        "experiment_name": "sample_train",
        "image_base_size": 600,
        "stride": 16,
        "grad_clip": 10,
        "bg_low": 0,
        "bg_high": 0.3,
        "fg_low": 0.7,
        "pos_prop_perc": 0.5,
        "prop_batch": 256,
        "pool_size": 7,
        "margin": 100,
        "clobber_positive": False,
        "neg_iou_thresh": 0.3,
        "pos_iou_thresh": 0.7,
        "pos_anchors_perc": 0.5,
        "anchor_batch": 256,
        "epochs": 100,
        "backbone": "resnet101",
        "detector_lr": 1e-4,
        "backbone_head_lr": 1e-4,
        "backbone_tail_lr": 1e-4,
        "rpn_lr": 1e-4,
        "train_type": "4step",
    },
}
