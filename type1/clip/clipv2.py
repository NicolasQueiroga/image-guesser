import os
import tensorflow as tf
import collections
import json
import numpy as np
from tqdm import tqdm


class PrepData:
    def __init__(self, datasets_dir: str = "datasets") -> None:
        tf.get_logger().setLevel("ERROR")
        self.root_dir = datasets_dir
        self.annotations_dir = os.path.join(self.root_dir, "annotations")
        self.images_dir = os.path.join(self.root_dir, "train2014")
        self.tfrecords_dir = os.path.join(self.root_dir, "tfrecords")
        self.annotation_file = os.path.join(self.annotations_dir, "captions_train2014.json")
        self.train_size = 30000
        self.valid_size = 5000
        self.captions_per_image = 2
        self.images_per_file = 2000

    def get_data(self, path: str) -> None:
        # Download caption annotation files
        if not os.path.exists(self.annotations_dir):
            annotation_zip = tf.keras.utils.get_file(
                "captions.zip",
                cache_dir=os.path.abspath("."),
                origin="http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                extract=True,
            )
            os.remove(annotation_zip)

        # Download image files
        if not os.path.exists(self.images_dir):
            image_zip = tf.keras.utils.get_file(
                "train2014.zip",
                cache_dir=os.path.abspath("."),
                origin="http://images.cocodataset.org/zips/train2014.zip",
                extract=True,
            )
            os.remove(image_zip)

        print("Dataset is downloaded and extracted successfully.")

    def build_coco(self) -> None:
        with open(self.annotation_file, "r") as f:
            annotations = json.load(f)["annotations"]
        
        self.image_path_to_caption = collections.defaultdict(list)
        for element in annotations:
            caption = f"{element['caption'].lower().rstrip('.')}"
            image_path = self.images_dir + "/COCO_train2014_" + "%012d.jpg" % (element["image_id"])
            self.image_path_to_caption[image_path].append(caption)

        self.image_paths = list(self.image_path_to_caption.keys())
        print(f"Number of images: {len(self.image_paths)}")

        self.train_image_paths = self.image_paths[:self.train_size]
        self.num_train_files = int(np.ceil(self.train_size / self.images_per_file))
        self.train_files_prefix = os.path.join(self.tfrecords_dir, "train")

        self.valid_image_paths = self.image_paths[-self.valid_size:]
        self.num_valid_files = int(np.ceil(self.valid_size / self.images_per_file))
        self.valid_files_prefix = os.path.join(self.tfrecords_dir, "valid")

        tf.io.gfile.makedirs(self.tfrecords_dir)
        
    def __bytes_feature(self, value: list) -> tf.train.Feature:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def __create_example(self, image_path: str, caption: str) -> tf.train.Example:
        feature = {
            "caption": self.bytes_feature(caption.encode()),
            "raw_image": self.bytes_feature(tf.io.read_file(image_path).numpy()),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    def __write_tfrecords(self, file_name: str) -> int:
        caption_list = []
        image_path_list = []
        for image_path in self.image_paths:
            captions = self.image_path_to_caption[image_path][:self.captions_per_image]
            caption_list.extend(captions)
            image_path_list.extend([image_path] * len(captions))

        with tf.io.TFRecordWriter(file_name) as writer:
            for example_idx in range(len(image_path_list)):
                example = self.create_example(
                    image_path_list[example_idx], caption_list[example_idx]
                )
                writer.write(example.SerializeToString())
        return example_idx + 1


    def write_data(self, image_paths: str, num_files: int, files_prefix: str) -> int:
        example_counter = 0
        for file_idx in tqdm(range(num_files)):
            file_name = files_prefix + "-%02d.tfrecord" % (file_idx)
            start_idx = self.images_per_file * file_idx
            end_idx = start_idx + self.images_per_file
            example_counter += self.write_tfrecords(file_name, image_paths[start_idx:end_idx])
        return example_counter


    def __read_example(self, example):
        features = tf.io.parse_single_example(
            example,
            {
                "caption": tf.io.FixedLenFeature([], tf.string),
                "raw_image": tf.io.FixedLenFeature([], tf.string),
            },
        )
        raw_image = features.pop("raw_image")
        features["image"] = tf.image.resize(
            tf.image.decode_jpeg(raw_image, channels=3), size=(299, 299)
        )
        return features


    def get_dataset(self, file_pattern, batch_size):

        return (
            tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
            .map(
                self.read_example,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            .shuffle(batch_size * 10)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .batch(batch_size)
        )