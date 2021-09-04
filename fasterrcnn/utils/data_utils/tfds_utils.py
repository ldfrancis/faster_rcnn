import tensorflow as tf
import tensorflow_datasets as tfds


def obtain_pascal_voc(split="train"):
    ds, info = tfds.load("voc", split=split, with_info=True, shuffle_files=True)
    return ds, info


def create_data(example, base_size):
    image = example["image"]
    H, W, C = image.shape
    bbox = example["objects"]["bbox"]
    y1, x1, y2, x2 = tf.unstack(bbox, 4, axis=1)
    x1 = x1 * (W - 1)
    y1 = y1 * (H - 1)
    x2 = x2 * (W - 1)
    y2 = y2 * (H - 1)
    label = example["objects"]["label"]
    bbox = tf.stack([x1, y1, x2, y2, tf.cast(label, tf.float32)], axis=-1)
    image, bbox = preprocess_data(image, bbox, base_size)
    return image, bbox


def modify_image_size(image, base_size):
    H, W, _ = image.shape
    SIZE = base_size
    aspect = W / H
    if W < H:
        new_W = SIZE
        new_H = int(SIZE / aspect)
    else:
        new_H = SIZE
        new_W = int(SIZE * aspect)

    image = tf.image.resize(image, [new_H, new_W])

    return image, new_H, new_W


def preprocess_data(image, bbox, base_size):
    H, W, _ = image.shape
    image, new_H, new_W = modify_image_size(image, base_size)

    x1, y1, x2, y2, c = tf.unstack(bbox, 5, axis=1)
    x1 = x1 * new_W / W
    y1 = y1 * new_H / H
    x2 = x2 * new_W / W
    y2 = y2 * new_H / H
    bbox = tf.stack([x1, y1, x2, y2, c], axis=1)

    return tf.cast(image, tf.int32), bbox
