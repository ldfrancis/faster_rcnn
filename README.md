# Faster RCNN: Step by Step

## Overview

Faster R-CNN is an object detection model that performs detection in 2 stages. The first stage proposes regions likely to contain objects, while the second stage detects actual objects in the proposed regions. The first stage is often referred to as the region proposal stage and the second, the detection stage. Region proposal is made possible by the use of a region proposal network, RPN. This network takes as input an image and produces bounding boxes for possible object regions in the image. A detector network is responsible for classifying each proposed region into the object class it contains. Both the RPN and the detector network rely on a pre-trained backbone network for image feature extraction. 

This implementation attempts to present all the components of faster R-CNN in simple steps that can serve as a reference for anyone on a quest to understand this object detection model. It is implemented in tensorflow.

## Installation
To install, just clone the repository and install with pip as illustrated below
git clone https://github.com/ldfrancis/faster_rcnn.git
cd faster_rcnn
pip install .

NB: You might find it helpful to use a virtual environment
## Usage
This model can be used either as a cli application or as a python package in a project.

### cli application
To use as a cli application, use any of the following commands to perform the desired action
1. Detect objects in an image

    `fasterrcnn --input ./input.jpg --output ./output_folder`
    
    This commad detects object in an image with the file path './input.jpg' and saves the result in the folder './output_folder'. The result includes an image having the bounding boxes of the detected objects drawn and json file containing detected objects information in terms of the bounding boxes and object classes.

    Also, the path to a folder containing several images can be supplied. In this case, objects would be detected in the each image and results would be saved in the supplied output folder.

    `fasterrcnn --input ./input_folder --output ./output_folder`

    If no output folder is supplied, the results would be saved in './output'

2. Train faster r-cnn on a dataset

    `fasterrcnn --train --dataset voc`

    This trains the model on the pascal voc dataset. To train on the coco dataset, you can specify 'coco'. To train on a custom dataset ...


### as a package
This can be used withing a python project like so;
```python
from fasterrcnn import FRCNN
from fasterrcnn.utils import (
    frcnn_default_config,
    load_config,
)
from fasterrcnn.data_utils import save_results

cfg = load_config("./path_to_config.yaml")
frcnn_default_config.update(cfg)

frcnn = FRCNN(frcnn_default_config)
image_path = "./image.jpg"
bboxes, scores = frcnn(image_path)

result = save_results(bboxes, scores)
```

For Training

```python
from fasterrcnn import FRCNN, Trainer
from fasterrcnn.config_utils import (
    frcnn_default_config,
    trainer_default_config,
    load_config,
)
from fasterrcnn.data_utils import save_results, obtain_dataset


cfg = load_config("./path_to_config.yaml")
frcnn_default_config.update(cfg)
trainer_default_config.update(cfg.get("trainer",{}))

frcnn = FRCNN(frcnn_default_config)
trainer = Trainer(frcnn, trainer_default_config)

train_dataset, val_dataset = obtain_dataset("name of dataset or path to dataset folder")

trainer.train(train_dataset, val_dataset)
```
. what is faster rcnn

. about this implementation

. Usage

. Results

- Step by Step Implementation

. Concept

. Arcihitecture / model structure

. Training

. Prediction






