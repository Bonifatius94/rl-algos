from typing import List, Union, Iterable

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras.layers import \
    Layer, Softmax, Lambda, Reshape


class STOneHotCategorical(Layer):
    def __init__(self, dims: Union[int, List[int]], name: str="st_cat_onehot"):
        super(STOneHotCategorical, self).__init__(name=name)
        self.softmax = Softmax()
        # self.softmax = Lambda(lambda x: tf.nn.log_softmax(x + 1e-8))
        self.sample = tfp.layers.OneHotCategorical(dims)
        self.stop_grad = Lambda(lambda x: tf.stop_gradient(x))

    def call(self, logits):
        samples = self.sample(logits)
        probs = self.softmax(logits)
        return samples + probs - self.stop_grad(probs)


class VQCodebook(Layer):
    """Representing a codebook of a vector quantization for a given amount
    of classifications with a given amount of classes each. The embedding
    vectors are initialized to match the inputs to be quantized. When calling
    this layer, it expects to receive one-hot encoded categoricals of shape
    (batch_size, num_classifications, num_classes)."""

    def __init__(
            self, num_classifications: int, num_classes: int,
            name: str="vq_codebook"):
        super(VQCodebook, self).__init__(name=name)
        self.num_classifications = num_classifications
        self.num_classes = num_classes
        self.num_embeddings = num_classifications * num_classes
        self.reshape_out = None

    def init_codebook(self, input_shape: Iterable[int]):
        orig_input_shape = input_shape[1:]
        self.reshape_out = Reshape(orig_input_shape)
        input_dims_flat = tf.reduce_prod(orig_input_shape)
        self.embedding_dims = input_dims_flat // self.num_classifications

        if input_dims_flat % self.num_classifications != 0:
            raise ValueError((
                f"The input dimensions {input_dims_flat} must be divisible "
                f"by the number of classifications {self.num_classifications} "
                f"to support swapping each of the {self.num_classifications} slices "
                "from the input vector with a quantized vector from the codebook."))

        embed_shape = (self.embedding_dims, self.num_embeddings)
        self.embeddings = self.add_weight(
            "embeddings", shape=embed_shape, trainable=True, initializer="random_normal")

    def call(self, categoricals_onehot: tf.Tensor):
        categoricals_sparse = tf.argmax(categoricals_onehot, axis=2)
        id_offsets = tf.range(0, self.num_classifications, dtype=tf.int64) * self.num_classes
        categoricals_embed_sparse = categoricals_sparse + id_offsets
        categoricals_embed = tf.one_hot(categoricals_embed_sparse, depth=self.num_embeddings)
        quantized = tf.matmul(categoricals_embed, self.embeddings, transpose_b=True)
        return self.reshape_out(quantized)

    def most_similar_embeddings(self, inputs: tf.Tensor):
        input_shape = (-1, self.num_classifications, self.embedding_dims)
        embed_shape = (-1, self.num_classifications, self.num_classes)
        inputs_per_classification = tf.reshape(inputs, input_shape)
        embeddings_per_classification = tf.reshape(self.embeddings, embed_shape)
        codebook_ids = []

        for i in range(self.num_classifications):
            embeddings = embeddings_per_classification[:, i, :]
            inputs_classif = inputs_per_classification[:, i, :]

            inputs_sqsum = tf.reduce_sum(inputs_classif ** 2, axis=1, keepdims=True)
            embed_sqsum = tf.reduce_sum(embeddings ** 2, axis=0)
            similarity = tf.matmul(inputs_classif, embeddings)
            distances = inputs_sqsum + embed_sqsum - 2 * similarity

            class_ids = tf.argmin(distances, axis=1, output_type=tf.int64)
            codebook_ids.append(tf.expand_dims(class_ids, axis=0))

        codebook_ids = tf.concat(codebook_ids, axis=0)
        codebook_ids = tf.transpose(codebook_ids, perm=[1, 0])
        return codebook_ids


class VQCategorical(Layer):
    """Representing a transformation of an input vector to be quantized into
    a one-hot encoded categorical matching the quantized vectors of the codebook.
    This layer can be used to receive a high-level latent state from arbitrary input.
    It expects to be used in combination with a codebook instance that is managing
    the embeddings used for quantization."""

    def __init__(self, codebook: VQCodebook, name: str="vq_categorical"):
        super(VQCategorical, self).__init__(name=name)
        self.codebook = codebook

    def build(self, input_shape: Iterable[int]):
        self.codebook.init_codebook(input_shape)

    def call(self, inputs: tf.Tensor):
        categoricals_sparse = self.codebook.most_similar_embeddings(inputs)
        return tf.one_hot(categoricals_sparse, self.codebook.num_classes)


class VQCombined(Layer):
    def __init__(
            self, num_classifications: int, num_classes: int,
            name: str="vq_combined"):
        super().__init__(name=name)
        self.vq_codebook = VQCodebook(num_classifications, num_classes, name=f"{name}_codebook")
        self.vq_categorical = VQCategorical(self.vq_codebook, name=f"{name}_categorical")

    def build(self, input_shape):
        self.vq_categorical.build(input_shape)

    def call(self, inputs):
        categoricals = self.vq_categorical(inputs)
        quantized = self.vq_codebook(categoricals)
        return quantized, categoricals
