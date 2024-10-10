from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class BiasCorrectionLayer(Layer):
    def __init__(self, initial_bias=0.0, **kwargs):
        super(BiasCorrectionLayer, self).__init__(**kwargs)
        self.initial_bias = initial_bias  # Store the initial bias value

    def build(self, input_shape):
        # Initialize the bias term with the provided initial bias value
        self.bias = self.add_weight(name='bias_correction',
                                    shape=(1,),
                                    initializer=tf.keras.initializers.Constant(self.initial_bias),
                                    trainable=True)

    def call(self, inputs):
        return inputs + self.bias