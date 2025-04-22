import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import StringLookup, Normalization
from tensorflow.keras.utils import Sequence

from keras import layers

from typing import List, Dict, Tuple


# Define the feature names and their types for the dataset
FEATURE_NAMES = {
    # "OBJECT_ID", # object
    "ISO0": "float32",
    "FITS_ID": "string",
    "FLAGS": "int16",
    "ELLIPTICITY": "float32",
    "CCD_ID": "uint8",
    "CLASS_STAR": "float32",
    "ELONGATION": "float32",
    "EXPTIME": "float32",
    "BACKGROUND": "float32"
}

NUMERIC_FEATURE_NAMES = {
    "ISO0": "float32",
    "BACKGROUND": "float32",
    "ELLIPTICITY": "float32",
    "ELONGATION": "float32",
    "CLASS_STAR": "float32",
    "EXPTIME": "float32"
}

CATEGORICAL_FEATURE_NAMES = {
    "FITS_ID": "string",
    "CCD_ID": "uint8",
    "FLAGS": "int16",
    # "gt_label1": "string" # is already part of the label
}

ID_COLUMNS = {
    "OBJECT_ID": "string"
}

# Defining the preprocessing layers for numeric and categorical features

class RobustNormalization(tf.keras.layers.Layer):
    """
    A custom Keras layer that performs robust normalization using the median and IQR.
    """
    def __init__(self, remove_outliers=False, **kwargs):
        super().__init__(**kwargs)
        self.built_stats = False
        self.remove_outliers = remove_outliers

    def adapt(self, dataset: tf.data.Dataset):
        """
        Compute the median and IQR (Interquartile Range) from a tf.data.Dataset.

        Args:
            dataset (tf.data.Dataset): Dataset yielding batches of features as tensors.
        """
        # Accumulate all batches into a single tensor.
        batches = []
        for batch in dataset:
            # If the dataset yields a tuple (x, y) then get x, otherwise assume it's the features.
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            batches.append(tf.cast(x, tf.float32))
        data = tf.concat(batches, axis=0)

        # Compute per-feature statistics along axis 0.
        self.median = tfp.stats.percentile(data, 50.0, interpolation='linear', axis=0)
        q1 = tfp.stats.percentile(data, 25.0, interpolation='linear', axis=0)
        q3 = tfp.stats.percentile(data, 75.0, interpolation='linear', axis=0)
        self.iqr = q3 - q1
        # Avoid division by zero when IQR is 0
        self.iqr = tf.where(tf.equal(self.iqr, 0.0), tf.ones_like(self.iqr), self.iqr)
        self.upper_bound = self.median + (1.5 * self.iqr)
        self.lower_bound = self.median - (1.5 * self.iqr)
        self.built_stats = True

    def call(self, inputs):
        if not self.built_stats:
            raise ValueError("The layer has not been adapted yet. Call 'adapt' with your training data.")
        if self.remove_outliers:
          inputs = tf.clip_by_value(inputs, self.lower_bound, self.upper_bound)
        return (inputs - self.median) / self.iqr


def get_normalization_layer(name: str, dataset: tf.data.Dataset) -> RobustNormalization:
    """
    Create a normalization layer for the specified feature.
    
    Args:
        name (str): The name of the feature to normalize.
        dataset (tf.data.Dataset): The dataset to adapt the normalization layer to.
    Returns:
        RobustNormalization: The normalization layer for the specified feature.
    """

    print("Processing numerical feature: ", name)
    # Create a Normalization layer for the feature.
    # if name == "ISO0" or name == "ELONGATION":
    normalizer = RobustNormalization()
    # else:
    #   normalizer = layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(name: str, 
                                dataset: tf.data.Dataset, 
                                dtype: str, 
                                max_tokens: int=None) -> tf.keras.layers.Layer:
    """
    Transform a categorical feature into an encoded vector.
    String and integer features are both converted to integer indices.

    Args:
        name (str): The name of the feature to encode.
        dataset (tf.data.Dataset): The dataset to adapt the encoding layer to.
        dtype (str): The data type of the feature ('string' or 'int').
        max_tokens (int): The maximum number of tokens for the StringLookup layer.
    
    Returns:
        tf.keras.layers.Layer: The encoding layer for the specified feature.
    """

    # Create a layer that turns strings into integer indices.
    print("Processing categorical feature: ", name)
    if dtype == 'string':
        print(" ... StringLookup")
        # Added oov_token to handle unseen values during training
        index = layers.StringLookup(max_tokens=max_tokens, oov_token='[UNK]')
    # Otherwise, create a layer that turns integer values into integer indices.
    else:
        print(" ... IntegerLookup")
        # Changed oov_token to -1 and added mask_token=None for IntegerLookup
        # mask_token=None to ensure that -1 index is not masked in CategoryEncoding
        index = layers.IntegerLookup(max_tokens=max_tokens, oov_token=-1, mask_token=None)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Encode the integer indices.
    # Added mask_zero=False to ensure the oov_token or -1 is not masked.
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))


def encode_inputs(dataset: tf.data.Dataset, 
                  feature_headers: Dict[str, str],
                  numeric_headers: Dict[str, str],
                  categorical_headers: Dict[str, str]) -> (dict, list):
    """
    Encode the inputs of the dataset using the specified feature headers.

    Args:
        dataset (tf.data.Dataset): The dataset to encode.
        feature_headers (dict): A dictionary of feature names and their types.
        numeric_headers (dict): A dictionary of numeric feature names and their types.
        categorical_headers (dict): A dictionary of categorical feature names and their types.
    Returns:
        tuple: A tuple containing a dictionary of all inputs and a list of encoded features.
    """

    all_inputs = {}
    encoded_features = []

    for feat_name, dtype in feature_headers.items():
        # Numerical features.
        if feat_name in numeric_headers.keys():
          numeric_col = tf.keras.Input(shape=(1,), name=feat_name)
          normalization_layer = get_normalization_layer(feat_name, dataset)
          encoded_numeric_col = normalization_layer(numeric_col)
          all_inputs[feat_name] = numeric_col
          encoded_features.append(encoded_numeric_col)
        elif feat_name in categorical_headers.keys():
          categorical_col = tf.keras.Input(shape=(1,), name=feat_name, dtype=dtype)
          max_tokens_value = 10 # Default value
          if feat_name == "FITS_ID": # Use a different value for FITS_ID
              max_tokens_value = 100 # Higher to accommodate more tokens
          encoding_layer = get_category_encoding_layer(name=feat_name,
                                                      dataset=dataset,
                                                      dtype=dtype,
                                                      max_tokens=max_tokens_value)
          encoded_categorical_col = encoding_layer(categorical_col)
          all_inputs[feat_name] = categorical_col
          encoded_features.append(encoded_categorical_col)


    return all_inputs, encoded_features

