import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, Conv2D, Concatenate, BatchNormalization, Activation, Flatten
from tensorflow.keras import Model

tf.keras.backend.set_floatx('float64')

# Actor model defined using Keras

class Q_Table:
    """Deep Q Model."""

    def __init__(self, state_size, action_size, name="Q_Table"):
        """Initialize parameters.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            name (string): Name of the model
        """
        # Initialize the state and action dimensions
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        
        # Build the actor model
        self.build_model()

    def build_model(self):
        ''' Build the Neural Net for Deep-Q learning Model.
        Convolutional layers with Batch Norm followed by dense layers.'''
        # Define input layer (states)
        states = Input(shape=self.state_size)

        # Input layer is 90x90x1
        net = Conv2D(16, (5, 5), strides=(2, 2), activation='relu', padding='same')(states)

        # Now 45x45x16
        net = Conv2D(32, (5, 5), strides=(2, 2), activation='relu', padding='same')(net)

        # Now 23x23x32
        net = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(net)
        # Now 12x12x64

        # Add dense layers
        net = Flatten()(net)
        net = Dense(units=128, activation='relu', kernel_initializer='glorot_uniform')(net)
        Q_values = Dense(units=self.action_size, activation='linear', kernel_initializer='glorot_uniform')(net)

        # Create Keras model
        self.model = Model(inputs=states, outputs=Q_values, name=self.name)

