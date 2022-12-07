import os
import json
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from tensorflow.keras import layers


class PrepData:
    def __init__(self, datasets_dir: str = "datasets") -> None:
        tf.get_logger().setLevel("ERROR")
        self.root_dir = datasets_dir
        self.annotations_dir = os.path.join(self.root_dir, "annotations")
        self.images_dir = os.path.join(self.root_dir, "train2014")
        self.tfrecords_dir = os.path.join(self.root_dir, "tfrecords")
        self.annotation_file = os.path.join(
            self.annotations_dir, "captions_train2014.json"
        )
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
            image_path = (
                self.images_dir
                + "/COCO_train2014_"
                + "%012d.jpg" % (element["image_id"])
            )
            self.image_path_to_caption[image_path].append(caption)

        self.image_paths = list(self.image_path_to_caption.keys())
        print(f"Number of images: {len(self.image_paths)}")

        self.train_image_paths = self.image_paths[: self.train_size]
        self.num_train_files = int(np.ceil(self.train_size / self.images_per_file))
        self.train_files_prefix = os.path.join(self.tfrecords_dir, "train")

        self.valid_image_paths = self.image_paths[-self.valid_size :]
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
            captions = self.image_path_to_caption[image_path][: self.captions_per_image]
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
            example_counter += self.write_tfrecords(
                file_name, image_paths[start_idx:end_idx]
            )
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


class PrepModel:
    def __init__(
        self,
        num_projection_layers: int,
        projection_dims: int,
        dropout_rate: float,
    ) -> None:
        self.num_projection_layers = num_projection_layers
        self.projection_dims = projection_dims
        self.dropout_rate = dropout_rate

    def project_embeddings(self):
        projected_embeddings = layers.Dense(units=self.projection_dims)(self.embeddings)
        for _ in range(self.num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = layers.Dense(self.projection_dims)(x)
            x = layers.Dropout(self.dropout_rate)(x)
            x = layers.Add()([projected_embeddings, x])
            projected_embeddings = layers.LayerNormalization()(x)
        return projected_embeddings

    def create_vision_encoder(self, trainable=False):
        # Load the pre-trained Xception model to be used as the base encoder.
        xception = tf.keras.applications.Xception(
            include_top=False, weights="imagenet", pooling="avg"
        )
        # Set the trainability of the base encoder.
        for layer in xception.layers:
            layer.trainable = trainable
        # Receive the images as inputs.
        inputs = layers.Input(shape=(299, 299, 3), name="image_input")
        # Preprocess the input image.
        xception_input = tf.keras.applications.xception.preprocess_input(inputs)
        # Generate the embeddings for the images using the xception model.
        embeddings = xception(xception_input)
        # Project the embeddings produced by the model.
        outputs = self.project_embeddings(embeddings)
        # Create the vision encoder model.
        return tf.keras.Model(inputs, outputs, name="vision_encoder")

    def create_text_encoder(self, trainable=False):
        # Load the BERT preprocessing module.
        preprocess = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2",
            name="text_preprocessing",
        )
        # Load the pre-trained BERT model to be used as the base encoder.
        bert = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
            name="bert",
        )
        # Set the trainability of the base encoder.
        bert.trainable = False  # trainable
        # Receive the text as inputs.
        inputs = layers.Input(shape=(), dtype=tf.string, name="text_input")
        # Preprocess the text.
        bert_inputs = preprocess(inputs)
        # Generate embeddings for the preprocessed text using the BERT model.
        embeddings = bert(bert_inputs)["pooled_output"]
        # Project the embeddings produced by the model.
        outputs = self.project_embeddings(embeddings)
        # Create the text encoder model.
        return tf.keras.Model(inputs, outputs, name="text_encoder")


class DualEncoder(tf.keras.Model):
    def __init__(self, text_encoder, image_encoder, temperature=1.0, **kwargs):
        super(DualEncoder, self).__init__(**kwargs)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False):
        # Place each encoder on a separate GPU (if available).
        # TF will fallback on available devices if there are fewer than 2 GPUs.
        with tf.device("/gpu:0"):
            # Get the embeddings for the captions.
            caption_embeddings = self.text_encoder(
                features["caption"], training=training
            )
        with tf.device("/gpu:1"):
            # Get the embeddings for the images.
            image_embeddings = self.vision_encoder(features["image"], training=training)
        return caption_embeddings, image_embeddings

    def compute_loss(self, caption_embeddings, image_embeddings):
        # logits[i][j] is the dot_similarity(caption_i, image_j).
        logits = (
            tf.matmul(caption_embeddings, image_embeddings, transpose_b=True)
            / self.temperature
        )
        # images_similarity[i][j] is the dot_similarity(image_i, image_j).
        images_similarity = tf.matmul(
            image_embeddings, image_embeddings, transpose_b=True
        )
        # captions_similarity[i][j] is the dot_similarity(caption_i, caption_j).
        captions_similarity = tf.matmul(
            caption_embeddings, caption_embeddings, transpose_b=True
        )
        # targets[i][j] = avarage dot_similarity(caption_i, caption_j) and dot_similarity(image_i, image_j).
        targets = tf.keras.activations.softmax(
            (captions_similarity + images_similarity) / (2 * self.temperature)
        )
        # Compute the loss for the captions using crossentropy
        captions_loss = tf.keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )
        # Compute the loss for the images using crossentropy
        images_loss = tf.keras.losses.categorical_crossentropy(
            y_true=tf.transpose(targets), y_pred=tf.transpose(logits), from_logits=True
        )
        # Return the mean of the loss over the batch.
        return (captions_loss + images_loss) / 2

    def train_step(self, features):
        with tf.GradientTape() as tape:
            # Forward pass
            caption_embeddings, image_embeddings = self(features, training=True)
            loss = self.compute_loss(caption_embeddings, image_embeddings)
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        caption_embeddings, image_embeddings = self(features, training=False)
        loss = self.compute_loss(caption_embeddings, image_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def train(
    pd: PrepData,
    epochs,
    batch_size,
    num_projection_layers=1,
    projection_dims=256,
    dropout_rate=0.1,
):
    # Load the data.
    pd.get_data()
    pd.build_coco_dataset()

    # Create vision and text encoders.
    pm = PrepModel(num_projection_layers, projection_dims, dropout_rate)
    vision_encoder = pm.create_vision_encoder()
    text_encoder = pm.create_text_encoder()

    # Create a dual encoder model.
    dual_encoder = DualEncoder(text_encoder, vision_encoder, temperature=1.0)

    # Compile the model.
    dual_encoder.compile(
        optimizer=tf.keras.optimizers.experimental.AdamW(
            learning_rate=0.001, weight_decay=0.001
        )
    )

    # set training and validation data.
    train_image_paths = pd.train_image_paths
    num_train_files = int(np.ceil(pd.train_size / pd.images_per_file))
    train_files_prefix = os.path.join(pd.tfrecords_dir, "train")
    train_example_count = pd.write_data(
        train_image_paths, train_files_prefix, num_train_files
    )
    print(f"{train_example_count} training examples were written to tfrecord files.")

    valid_image_paths = pd.valid_image_paths
    num_valid_files = int(np.ceil(pd.valid_size / pd.images_per_file))
    valid_files_prefix = os.path.join(pd.tfrecords_dir, "valid")
    valid_example_count = pd.write_data(
        valid_image_paths, valid_files_prefix, num_valid_files
    )
    print(f"{valid_example_count} evaluation examples were written to tfrecord files.")

    # Start training.
    print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"Number of examples (caption-image pairs): {train_example_count}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {int(np.ceil(train_example_count / batch_size))}")

    # Get TFRecord datasets.
    train_dataset = pd.get_dataset(
        os.path.join(pd.tfrecords_dir, "train-*.tfrecord"), batch_size
    )
    valid_dataset = pd.get_dataset(
        os.path.join(pd.tfrecords_dir, "valid-*.tfrecord"), batch_size
    )

    # Create a learning rate scheduler callback.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3
    )

    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Train the model.
    history = dual_encoder.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        callbacks=[reduce_lr, early_stopping],
    )
    print("Training completed. Saving vision and text encoders...")
    vision_encoder.save("vision_encoder")
    text_encoder.save("text_encoder")
    print("Models are saved.")

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "valid"], loc="upper right")
    plt.savefig("loss.png")

def read_image(image_path):
    image_array = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    return tf.image.resize(image_array, (299, 299))

def load_models():
    vision_encoder = tf.keras.models.load_model("vision_encoder")
    text_encoder = tf.keras.models.load_model("text_encoder")
    return vision_encoder, text_encoder
    
def generate_image_embeddings(image_paths, vision_encoder, batch_size):
    print(f"Generating embeddings for {len(image_paths)} images...")
    image_embeddings = vision_encoder.predict(
        tf.data.Dataset.from_tensor_slices(image_paths).map(read_image).batch(batch_size),
        verbose=1,
    )
    print(f"Image embeddings shape: {image_embeddings.shape}.")
    return image_embeddings

def find_matches(image_embeddings, image_paths, text_encoder, queries, k=9, normalize=True):
    # Get the embedding for the query.
    query_embedding = text_encoder(tf.convert_to_tensor(queries))
    # Normalize the query and the image embeddings.
    if normalize:
        image_embeddings = tf.math.l2_normalize(image_embeddings, axis=1)
        query_embedding = tf.math.l2_normalize(query_embedding, axis=1)
    # Compute the dot product between the query and the image embeddings.
    dot_similarity = tf.matmul(query_embedding, image_embeddings, transpose_b=True)
    # Retrieve top k indices.
    results = tf.math.top_k(dot_similarity, k).indices.numpy()
    # Return matching image paths.
    return [[image_paths[idx] for idx in indices] for indices in results]

def run_model(image_embeddings, image_paths, text_encoder, queries, k=9, normalize=True):
    matches = find_matches(
        image_embeddings, image_paths, text_encoder, queries, k, normalize
    )
    plt.figure(figsize=(20, 20))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(mpimg.imread(matches[i]))
        plt.axis("off")
    plt.show()
    return matches

def eval_accuracy(pd: PrepData, batch_size, image_embeddings, k=9):
        hits = 0
        num_batches = int(np.ceil(len(pd.image_paths) / batch_size))
        for idx in tqdm(range(num_batches)):
            start_idx = idx * batch_size
            end_idx = start_idx + batch_size
            current_image_paths = pd.image_paths[start_idx:end_idx]
            queries = [
                pd.image_path_to_caption[image_path][0] for image_path in current_image_paths
            ]
            result = find_matches(image_embeddings, queries, k)
            hits += sum(
                [
                    image_path in matches
                    for (image_path, matches) in list(zip(current_image_paths, result))
                ]
            )
        return hits / len(pd.image_paths)

    

def main(batch_size=128, epochs=10):
    pd = PrepData()
    if not os.path.exists("vision_encoder") or not os.path.exists("text_encoder"):
        train(pd=pd, batch_size=batch_size, epochs=epochs)
    
    # Load the models.
    vision_encoder, text_encoder = load_models()
    image_embeddings = generate_image_embeddings(pd.image_paths, vision_encoder, batch_size)

    # Get the queries.
    queries = ["a family standing next to the ocean on a sandy beach with a surf board"]
    matches = run_model(image_embeddings, pd.image_paths, text_encoder, queries)

    
    # Evaluate the accuracy.
    print("Scoring training data...")
    train_accuracy = eval_accuracy(pd, batch_size, image_embeddings)
    print(f"Train accuracy: {round(train_accuracy * 100, 3)}%")

    print("Scoring evaluation data...")
    eval_accuracy = eval_accuracy(pd, batch_size, image_embeddings)
    print(f"Eval accuracy: {round(eval_accuracy * 100, 3)}%")


if __name__ == "__main__":
    main()