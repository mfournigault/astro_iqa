import tensorflow as tf
from tf.keras import layers, Model
import tensorflow_recommenders as tfrs

from typing import Dict

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
