import time
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

# Adapted from jypyter notebook
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

# Create the FNO
class FourierLayer1D(Layer):
    def __init__(self, modes, **kwargs):
        super(FourierLayer1D, self).__init__(**kwargs)
        self.modes = modes

    def build(self, input_shape):
        self.time_steps = input_shape[1]  # Time dimension
        self.channels = input_shape[2]    # Channel dimension

        reg = tf.keras.regularizers.l2(1e-4)
        self.fourier_weights_real = self.add_weight(
            shape=(self.modes, self.channels, self.channels),
            initializer='glorot_uniform',
            trainable=True,
            name='fourier_weights_real',
            regularizer=reg
        )
        self.fourier_weights_imag = self.add_weight(
            shape=(self.modes, self.channels, self.channels),
            initializer='glorot_uniform',
            trainable=True,
            name='fourier_weights_imag',
            regularizer=reg
        )
        self.conv = Conv1D(self.channels, kernel_size=1, padding='same',
                           kernel_regularizer=reg)

    def call(self, inputs):
        # Compute FFT along the time dimension
        x_t = tf.transpose(inputs, perm=[0, 2, 1])
        x_ft = tf.signal.rfft(x_t, fft_length=[self.time_steps])
        # Truncate to first 'modes' frequency components
        x_ft_trunc = x_ft[:, :, :self.modes]
        x_ft_trunc = tf.transpose(x_ft_trunc, perm=[0, 2, 1])
        # Multiply real and imaginary parts with learned weights
        real_part = tf.einsum('bmc,mco->bmo', tf.math.real(x_ft_trunc), self.fourier_weights_real)
        imag_part = tf.einsum('bmc,mco->bmo', tf.math.imag(x_ft_trunc), self.fourier_weights_imag)
        x_ft_processed = tf.complex(real_part, imag_part)
        # Transpose back and pad to original frequency dimension
        x_ft_processed = tf.transpose(x_ft_processed, perm=[0, 2, 1])
        pad_size = x_ft.shape[-1] - self.modes
        x_ft_padded = tf.pad(x_ft_processed, [[0,0], [0,0], [0,pad_size]])
        # Inverse FFT to get back to time domain
        x_fourier = tf.signal.irfft(x_ft_padded, fft_length=[self.time_steps])
        x_fourier = tf.transpose(x_fourier, perm=[0, 2, 1])
        # Local convolution branch
        x_conv = self.conv(inputs)
        # Sum the Fourier branch and convolution branch
        return tf.keras.activations.relu(x_fourier + x_conv)

class FourierBlock1D(Layer):
    def __init__(self, modes, **kwargs):
        super(FourierBlock1D, self).__init__(**kwargs)
        self.fourier_layer = FourierLayer1D(modes)

    def call(self, inputs):
        x_skip = inputs
        x = self.fourier_layer(inputs)
        if x.shape == x_skip.shape:
            return tf.keras.activations.relu(x + x_skip)
        else:
            proj = Conv1D(x.shape[-1], kernel_size=1, padding='same')(x_skip)
            return tf.keras.activations.relu(x + proj)

def build_audio_fno(input_shape, num_classes, modes=32, num_blocks=4):
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=128, kernel_size=16, strides=2, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    # Stack a number of FourierBlock1D blocks.
    for _ in range(num_blocks):
        x = FourierBlock1D(modes=modes)(x)
        x = BatchNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

def prepare_data(df, max_length=None,  num_classes=10):
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

    # Process labels with fixed num_classes
    y = df['class'].values
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

param_grid = {
    'modes': [16, 32],
    'num_layers': [4, 6, 8, 10],
    'stride': [1, 2],
}

results = pd.DataFrame(columns=[
    'modes', 'num_layers', 'stride', 'dropout',
    'train_loss', 'val_loss', 'test_loss',
    'train_acc', 'val_acc', 'test_acc',
    'training_time', 'num_epochs'
])

if os.path.exists('tuning_results'):
    shutil.rmtree('tuning_results')
os.makedirs('tuning_results', exist_ok=True)

# Start hyperparameter search
for idx, params in enumerate(ParameterGrid(param_grid)):
    start_time = time.time()
    print(f"\nTraining combination {idx+1}/{len(ParameterGrid(param_grid))}: {params}")

    try:
        # Build model with current params
        def build_model():
            inputs = Input(shape=input_shape)
            x = Conv1D(128, 16, strides=params['stride'], padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(1e-3))(inputs)
            for _ in range(params['num_layers']):
                x = FourierLayer1D(modes=params['modes'])(x)
            x = GlobalAveragePooling1D()(x)
            x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
            outputs = Dense(10, activation='softmax')(x)
            return Model(inputs, outputs)

        model = build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'recall', 'F1Score']
        )

        filename = f"modes={params['modes']}_layers={params['num_layers']}_stride={params['stride']}.csv"

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                CSVLogger(f'tuning_results/{filename}')
            ]
        )

        # Evaluate
        train_loss, train_acc, train_recall, train_f1score = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc, val_recall, val_f1score = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc, test_recall, test_f1score = model.evaluate(X_test, y_test, verbose=0)

        # Store results in DataFrame
        results.loc[idx] = {
            'modes': params['modes'],
            'num_layers': params['num_layers'],
            'stride': params['stride'],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'test_recall': test_recall,
            'train_f1score': train_f1score,
            'val_f1score': val_f1score,
            'test_f1score': test_f1score,
            'training_time': time.time() - start_time,
            'num_epochs': len(history.history['loss'])
        }

        # Save incremental results
        results.to_csv('hyperparameter_results.csv', index=False)
        print(f"Saved results for combination {idx+1}")

        # Memory clean
        del model

    except Exception as e:
        print(f"Error training combination {params}: {str(e)}")
        continue

print("\nHyperparameter tuning complete!")
print("Final results saved to hyperparameter_results.csv")

from sklearn.model_selection import KFold
import numpy as np

hparam_results = pd.read_csv('hyperparameter_results.csv')
best_combo = hparam_results.loc[hparam_results['val_acc'].idxmax()]

print(f"\nBest Hyperparameters:")
print(f"• Modes: {best_combo['modes']}")
print(f"• Layers: {best_combo['num_layers']}")
print(f"• Stride: {best_combo['stride']}")

all_data = pd.concat([train_df, val_df, test_df])
kfold = KFold(n_splits=10)
num_classes = 10

fold_results = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(all_data)):
    print(f"\nProcessing fold {fold+1}/10")

    # Get fold data
    train_fold = all_data.iloc[train_idx]
    val_fold = all_data.iloc[val_idx]

    # Prepare data
    X_train, y_train, max_len = prepare_data(train_fold, num_classes=num_classes)
    X_val, y_val, _ = prepare_data(val_fold, max_length=max_len, num_classes=num_classes)

    # Build model with best hyperparameters
    def build_best_model(input_shape):
        inputs = Input(shape=input_shape)
        x = Conv1D(128, 16, strides=int(best_combo['stride']), padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-3))(inputs)
        for _ in range(int(best_combo['num_layers'])):
            x = FourierLayer1D(modes=int(best_combo['modes']))(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        return Model(inputs, outputs)

    model = build_best_model((max_len, 1))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'recall', 'F1Score'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
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
        'modes': best_combo['modes'],
        'layers': best_combo['num_layers'],
        'stride': best_combo['stride'],
    })

# Save and display results
kfold_results = pd.DataFrame(fold_results)
kfold_results.to_csv('best_combo_kfold_scores.csv', index=False)

print("\nK-fold Validation Results with Best Combo:")
print(kfold_results[['fold', 'val_acc', 'val_loss', 'val_recall', 'val_f1score', 'modes', 'layers', 'stride']])
print("\nSummary Statistics:")
print(kfold_results.describe())

import re

# Regex magic
def parse_filename(f):
    try:
        match = re.match(
            r'modes=(\d+)_layers=(\d+)_stride=(\d+).csv',
            os.path.basename(f)
        )
        if not match:
            return None

        return {
            'modes': int(match.group(1)),
            'layers': int(match.group(2)),
            'stride': int(match.group(3)),
        }
    except Exception as e:
        print(f"Error parsing {f}: {str(e)}")
        return None

param_files = []
for f in glob.glob("tuning_results/*.csv"):
    print(f"Processing: {f}")
    params = parse_filename(f)

    if not params:
        print(f"│── Skipping (invalid format)")
        continue

    try:
        df = pd.read_csv(f)
        df['modes'] = params['modes']
        df['layers'] = params['layers']
        df['stride'] = params['stride']
        param_files.append(df)
        print(f"└── Successfully loaded")
    except Exception as e:
        print(f"└── Loading error: {str(e)}")

if not param_files:
    print("\nNo valid files found. Verify filenames match pattern:")
    print("Found files:", os.listdir('tuning_results'))
    raise ValueError("No valid tuning data!")

full_df = pd.concat(param_files)

def plot_metric(full_df, metric='val_loss'):
    plt.figure(figsize=(15, 12))
    cmap = plt.get_cmap('plasma')

    # Plot by Fourier Modes
    plt.subplot(2, 2, 1)
    modes_unique = sorted(full_df['modes'].unique())
    for i, mode in enumerate(modes_unique):
        mask = full_df['modes'] == mode
        group = full_df[mask].groupby('epoch')[metric]
        color = cmap(i / len(modes_unique))
        plt.plot(group.mean(), label=f'Modes={mode}', color=color)
    plt.title(f'{metric} by Fourier Modes')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()

    # Plot by Number of Layers
    plt.subplot(2, 2, 2)
    layers_unique = sorted(full_df['layers'].unique())
    for i, layers in enumerate(layers_unique):
        mask = full_df['layers'] == layers
        group = full_df[mask].groupby('epoch')[metric]
        color = cmap(i / len(layers_unique))
        plt.plot(group.mean(), label=f'Layers={layers}', color=color)
    plt.title(f'{metric} by Number of Layers')
    plt.xlabel('Epochs')
    plt.grid(True)
    plt.legend()

    # Plot by Stride
    plt.subplot(2, 2, 3)
    stride_unique = sorted(full_df['stride'].unique())
    for i, stride in enumerate(stride_unique):
        mask = full_df['stride'] == stride
        group = full_df[mask].groupby('epoch')[metric]
        color = cmap(i / len(stride_unique))
        plt.plot(group.mean(), label=f'Stride={stride}', color=color)
    plt.title(f'{metric} by Stride')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()

    # Plot all hyperparameter combinations
    plt.subplot(2, 2, 4)
    combinations = list(full_df.groupby(['modes', 'layers', 'stride']))
    for i, ((modes, layers, stride), group) in enumerate(combinations):
        color = cmap(i / len(combinations))
        plt.plot(group[metric],
                 label=f'M{modes} L{layers} S{stride}',
                 color=color,
                 alpha=0.6,
                 linewidth=1)
    plt.title(f'All Hyperparameter Combinations ({metric})')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.yscale('log')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f'hyperparameter_analysis_{metric}.png', bbox_inches='tight', dpi=300)
    plt.show()

plot_metric(full_df, metric='val_loss')
plot_metric(full_df, metric='val_accuracy')
plot_metric(full_df, metric='loss')
plot_metric(full_df, metric='accuracy')

def plot_kfold_results(csv_path='best_combo_kfold_scores.csv'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x='fold', y='val_acc', marker='o', markersize=8)

    # Add trendline
    z = np.polyfit(df['fold'], df['val_acc'], 1)
    p = np.poly1d(z)
    plt.plot(df['fold'], p(df['fold']), 'r--',
            label=f'Trend (Slope: {z[0]:.2f})')

    plt.title('Validation Accuracy Across K-Folds')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig('kfold_trend.png', dpi=300)

plot_kfold_results()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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
    plt.savefig('class_wise_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

class_names = ['1', '2', '3', '4',
              '5', '6', '7', '8', '9', '0']

plot_class_accuracy(model, X_test, y_test, class_names)
