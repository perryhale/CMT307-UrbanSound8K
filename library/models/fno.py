import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class FourierLayer1D(Layer):
    def __init__(self, modes, **kwargs):
        super(FourierLayer1D, self).__init__(**kwargs)
        self.modes = modes

    def build(self, input_shape):
        self.time_steps = input_shape[1]  # Time dimension after downsampling
        self.channels = input_shape[2]    # Channel dimension

        # Frequency-domain weights
        self.fourier_weights_real = self.add_weight(
            shape=(self.modes, self.channels, self.channels),
            initializer='glorot_uniform',
            trainable=True,
            name='fourier_weights_real'
        )
        self.fourier_weights_imag = self.add_weight(
            shape=(self.modes, self.channels, self.channels),
            initializer='glorot_uniform',
            trainable=True,
            name='fourier_weights_imag'
        )

        # Local convolution
        self.conv = Conv1D(self.channels, 1, padding='same')

    def call(self, inputs):
        # FFT along time dimension
        x_t = tf.transpose(inputs, perm=[0, 2, 1])
        x_ft = tf.signal.rfft(x_t, fft_length=[self.time_steps])

        # Truncate to first 'modes' frequency components
        x_ft_trunc = x_ft[:, :, :self.modes]

        # Process real/imag parts with proper einsum
        x_ft_trunc = tf.transpose(x_ft_trunc, perm=[0, 2, 1])

        real_part = tf.einsum('bmc,mco->bmo',
                            tf.math.real(x_ft_trunc),
                            self.fourier_weights_real)
        imag_part = tf.einsum('bmc,mco->bmo',
                            tf.math.imag(x_ft_trunc),
                            self.fourier_weights_imag)
        x_ft_processed = tf.complex(real_part, imag_part)

        # Transpose back and pad
        x_ft_processed = tf.transpose(x_ft_processed, perm=[0, 2, 1])
        pad_size = x_ft.shape[-1] - self.modes
        x_ft_padded = tf.pad(x_ft_processed, [[0,0], [0,0], [0,pad_size]])

        # Inverse FFT
        x_fourier = tf.signal.irfft(x_ft_padded, fft_length=[self.time_steps])
        x_fourier = tf.transpose(x_fourier, perm=[0, 2, 1])

        # Local convolution branch
        x_conv = self.conv(inputs)

        return tf.keras.activations.relu(x_fourier + x_conv)

def build_audio_fno(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Downsampling for 96k samples
    x = Conv1D(128, 16, strides=2, padding='same')(inputs)  # 96000 → 48000

    # Fourier layers
    for _ in range(4):
        x = FourierLayer1D(modes=16)(x)

    # Classification head
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

def prepare_data(df, max_length=None):
    # Extract and normalize signals
    signals = []
    for s in df['data']:
        # Convert to float32 and normalize
        signal = np.array(s, dtype=np.float32)
        signal -= np.mean(signal)
        signal /= (np.std(signal) + 1e-8)
        signals.append(signal)

    # Pad/truncate to consistent length
    if max_length is None:
        max_length = max(len(s) for s in signals)

    X = pad_sequences(
        signals,
        maxlen=max_length,
        dtype='float32',
        padding='post',
        truncating='post'
    )

    # Add channel dimension
    X = X[..., np.newaxis]

    # Process labels
    y = df['class'].values
    num_classes = len(np.unique(y))
    y = tf.keras.utils.to_categorical(y, num_classes)

    return X, y, max_length

_, _, max_length = prepare_data(pd.concat([train_df, val_df, test_df]))

X_train, y_train, _ = prepare_data(train_df, max_length)
X_val, y_val, _ = prepare_data(val_df, max_length)
X_test, y_test, _ = prepare_data(test_df, max_length)

print(f"Shapes after preprocessing:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

input_shape = X_train.shape[1:]
print("Input shape:", input_shape)

test_input = tf.random.normal((2, 24000, 128))
layer = FourierLayer1D(modes=16)
test_output = layer(test_input)
print("Input shape:", test_input.shape)
print("Output shape:", test_output.shape)

model = build_audio_fno((96000, 1), 10)
model.summary()
