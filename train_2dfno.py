import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
import itertools
from sklearn.model_selection import ParameterGrid
import os
import shutil
import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
import glob
import seaborn as sns

from library.random import split_key
from library.data.io import reload_cache
from library.data.pipeline import (
	pad_and_slice_fn,
	cls_token_fn,
	expand_fn,
	expand_data,
	transform_data,
	partition_data,
	batch_generator,
	batch_signature
)
from library.data.descriptive import (
	wav_stats_fn,
	plot_distributions,
	plot_tokenized_sample
)

# Adapted from notebook
# read the csv file to a pandas df
df = reload_cache('urbansound8k_mono_24khz_float32.csv')
df.head()

plt.figure(figsize=(15, 6))
plt.plot(df['data'][0])
plt.title('Audio Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(['Signal'])
plt.xlim(0, len(df['data'][0]))
plt.show()

train_folds = [1,2,3,4,5,6]
val_folds = [7,8]
test_folds = [9, 10]

train_df = df[df['fold'].isin(train_folds)]
val_df = df[df['fold'].isin(val_folds)]
test_df = df[df['fold'].isin(test_folds)]

print("Shape of full data: ", df.shape)
print("Shape of training data: ", train_df.shape)
print("Shape of validation data: ", val_df.shape)
print("Shape of test data: ", test_df.shape)

class FourierLayer2D(Layer):
    def __init__(self, modes_x, modes_y, l2_reg=5e-4, **kwargs):
        super(FourierLayer2D, self).__init__(**kwargs)
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.input_channels = input_shape[-1]
        reg = tf.keras.regularizers.l2(self.l2_reg)
        self.weight_real = self.add_weight(
            shape=(self.modes_x, self.modes_y, self.input_channels, self.input_channels),
            initializer='glorot_normal',
            trainable=True,
            name='fourier_weight_real',
            regularizer=reg
        )
        self.weight_imag = self.add_weight(
            shape=(self.modes_x, self.modes_y, self.input_channels, self.input_channels),
            initializer='glorot_normal',
            trainable=True,
            name='fourier_weight_imag',
            regularizer=reg
        )

        self.conv = Conv2D(
            filters=self.input_channels,
            kernel_size=1,
            padding='same',
            kernel_regularizer=reg
        )

    def call(self, inputs):
        # Combine weights into a complex tensor
        weights = tf.complex(self.weight_real, self.weight_imag)
        # Compute 2D FFT
        x_ft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))
        x_ft_trunc = x_ft[:, :self.modes_x, :self.modes_y, :]
        # Apply learned Fourier weights using einsum
        x_ft_processed = tf.einsum('bxyi,xyio->bxyo', x_ft_trunc, weights)
        pad_x = tf.shape(x_ft)[1] - self.modes_x
        pad_y = tf.shape(x_ft)[2] - self.modes_y
        x_ft_padded = tf.pad(x_ft_processed, [[0, 0], [0, pad_x], [0, pad_y], [0, 0]])
        # Inverse FFT to return to spatial domain
        x_ifft = tf.signal.ifft2d(x_ft_padded)
        x = tf.math.real(x_ifft)
        x_local = self.conv(inputs)
        return tf.keras.activations.relu(x + x_local)

# Block that wraps FourierLayer2D and adds a residual connection.
class FourierBlock2D(Layer):
    def __init__(self, modes_x, modes_y, l2_reg=1e-4, **kwargs):
        super(FourierBlock2D, self).__init__(**kwargs)
        self.fourier_layer = FourierLayer2D(modes_x, modes_y, l2_reg)

    def call(self, inputs):
        x = self.fourier_layer(inputs)
        if x.shape == inputs.shape:
            return tf.keras.activations.relu(x + inputs)
        else:
            proj = Conv2D(filters=x.shape[-1], kernel_size=1, padding='same')(inputs)
            return tf.keras.activations.relu(x + proj)

def build_spectrogram_fno(input_shape, num_classes,
                          modes_x=16, modes_y=32,
                          num_layers=4, hidden_dim=32,
                          l2_reg=5e-4):
    inputs = Input(shape=input_shape)
    x = inputs

    x = Conv2D(hidden_dim, kernel_size=3, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    for _ in range(num_layers):
        x = FourierBlock2D(modes_x=min(modes_x, input_shape[0]),
                           modes_y=min(modes_y, input_shape[1]),
                           l2_reg=l2_reg)(x)
        x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

input_shape = (186, 128, 1)
model = build_spectrogram_fno(input_shape, num_classes=10)
model.summary()

test_input = tf.random.normal([2, 186, 128, 1])
test_output = model(test_input)
print("Test output shape:", test_output.shape)

def create_mel_spectrogram(waveform, target_length=96000, frame_length=1024,
                           frame_step=512, n_mels=128, sr=16000):
    # Pad/trim waveform to fixed length
    if len(waveform) > target_length:
        waveform = waveform[:target_length]
    else:
        waveform = np.pad(waveform, (0, max(0, target_length - len(waveform))))

    # Cast waveform to float32
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    # Compute STFT
    stft = tf.signal.stft(
        waveform,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    magnitude = tf.abs(stft)

    # Create Mel filter bank
    linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=frame_length // 2 + 1,
        sample_rate=sr,
        lower_edge_hertz=0.0,
        upper_edge_hertz=sr/2
    )

    # Apply Mel scaling
    mel_spectrogram = tf.tensordot(magnitude, linear_to_mel_matrix, axes=1)
    log_mel = tf.math.log(mel_spectrogram + 1e-6)
    return log_mel.numpy()[..., tf.newaxis]

def prepare_spectrograms(df, target_length=96000, frame_length=1024,
                        frame_step=512, n_mels=128, sr=16000):
    # Calculate fixed dimensions
    n_time = (target_length - frame_length) // frame_step + 1
    fixed_dims = (n_time, n_mels)

    spectrograms = []
    valid_indices = []

    for idx, waveform in enumerate(df['data']):
        try:
            spec = create_mel_spectrogram(
                waveform,
                target_length=target_length,
                frame_length=frame_length,
                frame_step=frame_step,
                n_mels=n_mels,
                sr=sr
            )
            if spec.shape[:2] == fixed_dims:
                spectrograms.append(spec)
                valid_indices.append(idx)
        except Exception as e:
            print(f"Skipping sample {idx}: {str(e)}")

    labels = df.iloc[valid_indices]['class'].values
    return np.array(spectrograms), tf.keras.utils.to_categorical(labels, num_classes=10)

params = {
    'target_length': 96000,
    'frame_length': 1024,
    'frame_step': 512,
    'n_mels': 128,
    'sr': 16000
}

X_train_s, y_train_s = prepare_spectrograms(train_df, **params)
X_val_s, y_val_s = prepare_spectrograms(val_df, **params)
X_test_s, y_test_s = prepare_spectrograms(test_df, **params)

print("\nMel Spectrogram Shapes:")
print(f"Train: {X_train_s.shape}, y: {y_train_s.shape}")
print(f"Val: {X_val_s.shape}, y: {y_val_s.shape}")
print(f"Test: {X_test_s.shape}, y: {y_test_s.shape}")

# Update model construction for Mel spectrogram input
input_shape = X_train_s.shape[1:]
print("Input shape:", input_shape)

# Test 2D Fourier layer with Mel dimensions
test_input = tf.random.normal((2, 186, 128, 1))
layer = FourierLayer2D(modes_x=16, modes_y=16)
test_output = layer(test_input)
print("Test input shape:", test_input.shape)
print("Test output shape:", test_output.shape)

model = build_spectrogram_fno(input_shape, num_classes=10)
model.summary()

def plot_waveform_and_mel(waveform, sr, mel_spec):

    time_axis = np.arange(len(waveform)) / sr

    mel_image = np.squeeze(mel_spec, axis=-1).T
    mel_image = np.flipud(mel_image)

    n_time, n_mels = mel_spec.shape[:2]
    duration = (n_time - 1) * (1024/2) / sr
    freq_extent = sr / 2

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot the waveform
    axs[0].plot(time_axis, waveform, color='gray')
    axs[0].set_title("Waveform")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")

    im = axs[1].imshow(mel_image, aspect='auto', origin='lower',
                       extent=[0, duration, 0, freq_extent],
                       cmap='plasma')
    axs[1].set_title("Log-Mel Spectrogram")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axs[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()

sr = 16000
target_length = 96000
waveform = test_df['data'].iloc[0]

mel_spec = create_mel_spectrogram(waveform, target_length=target_length, frame_length=1024,
                                  frame_step=512, n_mels=128, sr=sr)

plot_waveform_and_mel(waveform, sr, mel_spec)

import time

# 2D parameter grid
param_grid = {
    'modes_x': [12, 24],      # Time dimension modes
    'modes_y': [32, 64],      # Frequency dimension modes
    'num_layers': [4, 8],
    'hidden_dim': [32, 64],
    'learning_rate': [1e-3],
    'batch_size': [16, 32],
}

results = pd.DataFrame(columns=[
    'modes_x', 'modes_y', 'num_layers', 'hidden_dim',
    'learning_rate', 'batch_size',
    'train_loss', 'val_loss', 'test_loss',
    'train_acc', 'val_acc', 'test_acc',
    'training_time', 'num_epochs'
])

# Directory setup
if os.path.exists('2d_tuning_results'):
    shutil.rmtree('2d_tuning_results')
os.makedirs('2d_tuning_results', exist_ok=True)

def train_and_evaluate_model(params):
    start_time = time.time()

    # Build model
    model = build_spectrogram_fno(
        input_shape=X_train_s.shape[1:],
        num_classes=10,
        modes_x=params['modes_x'],
        modes_y=params['modes_y'],
        num_layers=params['num_layers'],
        hidden_dim=params['hidden_dim'],
    )

    # Compile with dynamic learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'recall', 'F1Score']
    )

    # Create parameter string
    param_str = "_".join([f"{k}={v}" for k,v in params.items()])

    # Training with callbacks
    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=50,
        batch_size=params['batch_size'],
        verbose=0,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            CSVLogger(f'2d_tuning_results/{param_str}.csv')
        ]
    )

    # Evaluation
    train_loss, train_acc, train_recall, train_f1score = model.evaluate(X_train_s, y_train_s, verbose=0)
    val_loss, val_acc, val_recall, val_f1score = model.evaluate(X_val_s, y_val_s, verbose=0)
    test_loss, test_acc, test_recall, test_f1score = model.evaluate(X_test_s, y_test_s, verbose=0)

    # Memory clean
    del model

    return {
        **params,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'val_recall': val_recall,
        'test_recall': test_recall,
        'val_f1score': val_f1score,
        'test_f1score': test_f1score,
        'training_time': time.time() - start_time,
        'num_epochs': len(history.history['loss'])
    }

# Main training loop
for idx, params in enumerate(ParameterGrid(param_grid)):
    print(f"\nTraining combination {idx+1}/{len(ParameterGrid(param_grid))}: {params}")

    try:
        result = train_and_evaluate_model(params)
        results.loc[idx] = result

        # Save incremental results
        results.to_csv('2d_hyperparameter_results.csv', index=False)
        print(f"Saved results for combination {idx+1}")

    except Exception as e:
        print(f"Error training combination {params}: {str(e)}")
        continue

print("\n2D Hyperparameter tuning complete!")
print("Final results saved to 2d_hyperparameter_results.csv")

hparam_results = pd.read_csv('2d_hyperparameter_results.csv')
best_combo = hparam_results.loc[hparam_results['val_acc'].idxmax()]

print(f"\nBest Hyperparameters:")
print(f"• Modes x: {best_combo['modes_x']}")
print(f"• Modes y: {best_combo['modes_y']}")
print(f"• Num of layers: {best_combo['num_layers']}")

all_data = pd.concat([train_df, val_df, test_df])
kfold = KFold(n_splits=10)
num_classes = 10

spect_params = {
    'target_length': 96000,
    'frame_length': 1024,
    'frame_step': 512,
    'n_mels': 128,
    'sr': 16000
}

def prepare_spectrograms(df, params):
    specs = []
    labels = []

    for idx, waveform in enumerate(df['data']):
        try:
            # Generate Mel spectrogram
            spec = create_mel_spectrogram(
                waveform,
                target_length=params['target_length'],
                frame_length=params['frame_length'],
                frame_step=params['frame_step'],
                n_mels=params['n_mels'],
                sr=params['sr']
            )

            # Force fixed dimensions with padding
            padded_spec = np.pad(
                spec,
                ((0, 186 - spec.shape[0]), (0, 0), (0, 0)),
                mode='constant'
            )
            specs.append(padded_spec)
            labels.append(df.iloc[idx]['class'])

        except Exception as e:
            print(f"Skipping sample {idx}: {str(e)}")

    return np.array(specs), tf.keras.utils.to_categorical(labels, num_classes=10)

fold_results = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(all_data)):
    print(f"\nProcessing fold {fold+1}/10")

    # Get fold data
    train_fold = all_data.iloc[train_idx]
    val_fold = all_data.iloc[val_idx]

    # Prepare data
    X_train_s, y_train_s = prepare_spectrograms(train_fold, spect_params)
    X_val_s, y_val_s = prepare_spectrograms(val_fold, spect_params)


    print(f"Train shape: {X_train_s.shape}, Val shape: {X_val_s.shape}")

    # Build model with best hyperparameters
    model_s = build_spectrogram_fno(
        input_shape=(186, 128, 1),
        num_classes=10,
        modes_x=int(best_combo['modes_x']),
        modes_y=int(best_combo['modes_y']),
        num_layers=int(best_combo['num_layers']),
        hidden_dim=int(best_combo['hidden_dim'])
    )

    model_s.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'recall', 'F1Score'])

    history = model_s.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=[EarlyStopping(patience=5)]
    )

    # Record results
    best_epoch = np.argmax(history.history['val_accuracy'])
    fold_results.append({
        'fold': fold + 1,
        'val_loss': history.history['val_loss'][best_epoch],
        'val_acc': history.history['val_accuracy'][best_epoch],
        'val_recall': history.history['val_recall'][best_epoch],
        'val_f1score': history.history['val_F1Score'][best_epoch],
        'layers': best_combo['num_layers'],
    })

# Save and display results
kfold_results = pd.DataFrame(fold_results)
kfold_results.to_csv('2d_best_combo_kfold_scores.csv', index=False)

print("\nK-fold Validation Results with Best Combo:")
print(kfold_results[['fold', 'val_acc', 'val_loss', 'val_recall', 'val_f1score', 'layers']])
print("\nSummary Statistics:")
print(kfold_results.describe())

import re

# More regex magic
def parse_2d_filename(f):
    try:
        pattern = (
            r'batch_size=(\d+)_'
            r'hidden_dim=(\d+)_'
            r'learning_rate=([\d.e-]+)_'
            r'modes_x=(\d+)_'
            r'modes_y=(\d+)_'
            r'num_layers=(\d+)'
        )
        match = re.match(pattern, os.path.basename(f))

        if not match:
            return None

        return {
            'batch_size': int(match.group(1)),
            'hidden_dim': int(match.group(2)),
            'learning_rate': float(match.group(3)),
            'modes_x': int(match.group(4)),
            'modes_y': int(match.group(5)),
            'num_layers': int(match.group(6))
        }
    except Exception as e:
        print(f"Error parsing {f}: {str(e)}")
        return None

def load_2d_results(results_dir="2d_tuning_results"):
    param_files = []
    for f in glob.glob(f"{results_dir}/*.csv"):
        print(f"Processing: {f}")
        params = parse_2d_filename(f)

        if not params:
            print(f"│── Skipping (invalid format)")
            continue

        try:
            df = pd.read_csv(f)
            # Add hyperparameters as columns
            for k, v in params.items():
                df[k] = v
            param_files.append(df)
            print(f"└── Successfully loaded")
        except Exception as e:
            print(f"└── Loading error: {str(e)}")

    if not param_files:
        raise ValueError("No valid 2D tuning data found!")

    return pd.concat(param_files)

def plot_all_combinations(results_dir="2d_tuning_results", metric='val_accuracy'):
    plt.figure(figsize=(15, 12))
    cmap = plt.get_cmap('plasma')

    files = glob.glob(f"{results_dir}/*.csv")
    combinations = []

    # Plot all lines first
    for idx, f in enumerate(files):
        params = parse_2d_filename(f)
        if not params:
            continue

        df = pd.read_csv(f)
        label = (f"mx{params['modes_x']}-my{params['modes_y']} "
                 f"ly{params['num_layers']}-hd{params['hidden_dim']} "
                 f"lr{params['learning_rate']:.0e}-bs{params['batch_size']}")

        plt.plot(df['epoch'], df[metric],
                 color=cmap(idx/len(files)),
                 alpha=0.7,
                 linewidth=1.5,
                 label=label)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Set labels and title
    plt.title(f"{metric} Development by Epoch", pad=20)
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())

    # Configure legend
    legend = plt.legend(
        bbox_to_anchor=(0.5, -0.2),
        loc='upper center',
        ncol=4,  # Number of columns in legend
        fontsize=8,
        frameon=False
    )

    # Adjust layout to make space for legend
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Add grid to background
    plt.gca().set_axisbelow(True)

    # Log scale for loss metrics
    if 'loss' in metric:
        plt.yscale('log')

    plt.savefig(f'all_combinations_{metric}.png',
                dpi=300,
                bbox_extra_artists=(legend,),
                bbox_inches='tight')
    plt.show()

plot_all_combinations(metric='val_accuracy')
plot_all_combinations(metric='val_loss')
plot_all_combinations(metric='accuracy')
plot_all_combinations(metric='loss')

import plotly.express as px

def interactive_plot(results_dir="2d_tuning_results", metric='val_accuracy'):
    dfs = []
    for f in glob.glob(f"{results_dir}/*.csv"):
        params = parse_2d_filename(f)
        if not params:
            continue
        df = pd.read_csv(f)
        for k, v in params.items():
            df[k] = v
        dfs.append(df)

    full_df = pd.concat(dfs)

    fig = px.line(full_df,
                 x='epoch',
                 y=metric,
                 color='batch_size',
                 line_dash='num_layers',
                 facet_col='modes_x',
                 facet_row='modes_y',
                 hover_data=['hidden_dim', 'learning_rate'],
                 title=f"{metric} by Epoch")

    fig.update_layout(height=800, width=1200)
    fig.show()
    fig.write_html(f"interactive_{metric}.html")

interactive_plot(metric='val_accuracy')
interactive_plot(metric='accuracy')

def plot_class_accuracy(model, X_test, y_test, class_names):
    # Convert one-hot encoded labels back to integers
    y_true = np.argmax(y_test, axis=1)

    # Get predictions and convert to class indices
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Extract class-wise accuracy
    class_acc = {class_names[int(k)]: v['recall']
                 for k, v in report.items() if k.isdigit()}

    # Create dataframe for plotting
    acc_df = pd.DataFrame.from_dict(class_acc, orient='index', columns=['Accuracy'])
    acc_df = acc_df.reset_index().rename(columns={'index': 'Class'})

    # Create plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Class', y='Accuracy', data=acc_df, palette='viridis')

    # Add annotations
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')

    plt.title('Classification Accuracy per Class', pad=20, fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('class_wise_accuracy_2d.png', dpi=300, bbox_inches='tight')
    plt.show()

#class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
#              'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plot_class_accuracy(model_s, X_test_s, y_test_s, class_names)

from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, X_test, y_test, class_names):
    # Convert one-hot encoded labels back to integers
    y_true = np.argmax(y_test, axis=1)

    # Get predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create DataFrame for easier plotting
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cbar=False)

    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_2d.png', dpi=300, bbox_inches='tight')
    plt.show()

# Usage with your actual class names (uncomment and replace with your real labels)
class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
              'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']


plot_confusion_matrix(model_s, X_test_s, y_test_s, class_names)
