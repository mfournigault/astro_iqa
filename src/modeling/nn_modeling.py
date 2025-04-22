import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_recommenders as tfrs

from typing import List, Dict # Importing List from typing module

### DEEP CROSS NETWORK (DCN) MODEL ###
# The DCN model is a combination of a DNN and a cross network.
# The DNN learns the low-order feature interactions, while the cross network learns the high-order feature interactions.
# The DCN model is a generalization of the Factorization Machine (FM) model.
def create_dcn_model(all_inputs: Dict[str, tf.keras.layers.Input],
                 encoded_inputs: List[tf.keras.layers.Layer],
                 num_hidden_layers: int=2, 
                 units_per_layer: int=64,
                 num_cross_layers: int=1,
                 dcn_dnn: str="stack",
                 dropout_rate: float=0.15, 
                 l2: float=0.01) -> tf.keras.Model:
    """
    Create a Deep Croos Network model with keras preprocessing layers.
    The preprocessing layers take care of normalising the numerical features and encoding the categorical features.
    The model is built using the Keras functional API.

    The model is a Deep Cross Network (DCN) with a specified number of hidden layers and cross layers.
    The cross layers are DCN V2 from the TRFS library.
    The DNN layers are either stacked on top of the cross layers or concatenated with the cross layers.
    """
    all_features = layers.concatenate(encoded_inputs)

    cross_layer = tfrs.layers.dcn.Cross(projection_dim=None,
                                        kernel_initializer="glorot_uniform",
                                        kernel_regularizer=tf.keras.regularizers.l2(l2),
                                        bias_regularizer=tf.keras.regularizers.l2(l2))

    # According to the publicaiton, with more than 2 cross layers, performances are not improved that much
    x = all_features
    for i in range(num_cross_layers):
        x = cross_layer(all_features, x)

    # we stack the DNN on top of the cross layer, but we can also concatenate the outputs of the cross layer and the DNN
    dcn_outputs = layers.BatchNormalization()(x)
    if dcn_dnn == "stack":
        dnn_inputs = dcn_outputs
    elif dcn_dnn == "concatenate":
        dnn_inputs = all_features
    else:
        raise ValueError("dcn_dnn must be either 'stack' or 'concatenate'")

    x = dnn_inputs
    for i in range(num_hidden_layers, 0, -1):
        num_units = units_per_layer * i
        x = layers.Dense(units=num_units,
                         activation="relu",
                         kernel_regularizer=tf.keras.regularizers.l2(l2),
                         bias_regularizer=tf.keras.regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

    if dcn_dnn == "stack":
        softmax_inputs = x
    else:  # we managed unkown cases earlier
        softmax_inputs = layers.concatenate([dcn_outputs, x])

    # outputs = layers.Dense(units=5, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(l2), bias_regularizer=tf.keras.regularizers.l2(l2))(x)
    outputs = layers.Dense(units=5, activation="sigmoid")(softmax_inputs)

    # Create the model.
    model = Model(inputs=all_inputs, outputs=outputs)

    return model


### DEEP NEURAL NETWORK (DNN) MODEL ###
# The DNN model is a simple feedforward neural network with a specified number of hidden layers.
# The DNN model is a generalization of the Multi-Layer Perceptron (MLP) model.
def create_dnn_model(
        all_inputs: Dict[str, tf.keras.layers.Input],
        encoded_inputs: List[tf.keras.layers.Layer],
        num_hidden_layers: int=2, 
        units_per_layer: int=64,
        dropout_rate: float=0.15, 
        l2: float=0.01) -> tf.keras.Model:
    """
    Create a model with embedding layers for categorical features.
    """
    all_features = layers.concatenate(encoded_inputs)
    x = layers.BatchNormalization()(all_features)
    for i in range(num_hidden_layers, 0, -1):
        num_units = units_per_layer * i
        x = layers.Dense(units=num_units,
                         activation="relu",
                         kernel_regularizer=tf.keras.regularizers.l2(l2),
                         bias_regularizer=tf.keras.regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(units=5, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(l2), bias_regularizer=tf.keras.regularizers.l2(l2))(x)
    model = Model(inputs=all_inputs, outputs=outputs)

    return model


### GRN / VSN MODEL ###
# The GRN model is a simple feedforward neural network with a specified number of hidden layers.
# The GRN is based on Gated Linear Unit.
## Implement the Gated Linear Unit

# [Gated Linear Units (GLUs)](https://arxiv.org/abs/1612.08083) provide the
# flexibility to suppress input that are not relevant for a given task.

class GatedLinearUnit(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

## Implement the Gated Residual Network

# The Gated Residual Network (GRN) works as follows:

# 1. Applies the nonlinear ELU transformation to the inputs.
# 2. Applies linear transformation followed by dropout.
# 4. Applies GLU and adds the original inputs to the output of the GLU to perform skip
# (residual) connection.
# 6. Applies layer normalization and produces the output.

class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate):
        super().__init__()
        self.units = units
        self.elu_dense = layers.Dense(units, activation="elu")
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x

## Implement the Variable Selection Network

# The Variable Selection Network (VSN) works as follows:

# 1. Applies a GRN to each feature individually.
# 2. Applies a GRN on the concatenation of all the features, followed by a softmax to
# produce feature weights.
# 3. Produces a weighted sum of the output of the individual GRN.

# Note that the output of the VSN is [batch_size, encoding_size], regardless of the
# number of the input features.


class VariableSelection(layers.Layer):
    def __init__(self, num_features, units, dropout_rate):
        super().__init__()
        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = layers.Dense(units=num_features, activation="softmax")

    def call(self, inputs):
        # MFT: in our case, we will use (or try at least) to use the categorical features without
        # embeddings, as they don't have a high dimensionality. One represent the file name, another the CCD_ID between
        # 1 and 16, the target represents 5 different classes, and the flags can take few values as well.
        v = layers.concatenate(inputs)
        v = self.grn_concat(v)
        # v = keras.ops.expand_dims(self.softmax(v), axis=-1)
        v = tf.expand_dims(self.softmax(v), axis=-1)

        x = []
        for idx, input in enumerate(inputs):
            x.append(self.grns[idx](input))
        # x = keras.ops.stack(x, axis=1)
        x = tf.stack(x, axis=1)

        # outputs = keras.ops.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        return outputs
