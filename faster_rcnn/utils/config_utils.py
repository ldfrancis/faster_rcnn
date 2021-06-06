from datetime import datetime
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
