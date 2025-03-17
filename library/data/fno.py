import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Activation, Conv2D
from tensorflow.keras.models import Model

class FourierLayer(Layer):
    def __init__(self, modes, **kwargs):
        """
        modes: number of low-frequency modes to keep along each axis.
        """
        super(FourierLayer, self).__init__(**kwargs)
        self.modes = modes

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.weights_complex = self.add_weight(
            shape=(self.modes, self.modes, channels, channels),
            initializer='glorot_uniform',
            trainable=True,
            name="weights_complex"
        )
        super(FourierLayer, self).build(input_shape)

    def __call__(self, inputs):
        # Compute the 2D FFT of the input
        input_fft = tf.signal.rfft2d(inputs)
        # Get the input shape
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # Extract low-frequency modes along height and width
        input_fft_low = input_fft[:, :self.modes, :self.modes, :]

        output_fft_low = tf.einsum("bxyc,xycd->bxy d", input_fft_low, self.weights_complex)
        
        zeros = tf.zeros_like(input_fft)
        output_fft = tf.concat([
            tf.concat([output_fft_low, zeros[:, :self.modes, self.modes:, :]], axis=2),
            zeros[:, self.modes:, :, :]
        ], axis=1)

        # Apply the inverse FFT to convert back to the spatial domain.
        output = tf.signal.irfft2d(output_fft, fft_length=[height, width])
        return output
"""
# Example usage
input_tensor = Input(shape=(64, 64, 3))
x = FourierLayer(modes=16)(input_tensor)
x = tf.keras.layers.Activation("relu")(x)
output_tensor = tf.keras.layers.Conv2D(3, kernel_size=1)(x)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.summary()
"""
