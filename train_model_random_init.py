import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks

import time
import random
import pickle

from library.random import split_key
from library.data.io import reload_cache
from library.data.pipeline import (
	pad_and_slice_fn,
	cls_token_fn,
	expand_fn,
	expand_data,
	transform_data,
	partition_data,
	dataset_generator,
	dataset_signature
)
from library.data.descriptive import (
	wav_stats_fn,
	plot_distributions,
	plot_tokenized_sample
)
from library.models.transformer import get_denoising_transformer_encoder, convert_dte_to_classifier


### setup

# start timer
T0 = time.time()
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# init RNG seeds
K0 = 999
K1, K2 = split_key(K0)

# set global RNG seed
random.seed(K1)
np.random.seed(K1)
tf.random.set_seed(K1)


### hyperparameters

# architecture
N_TOKENS = 512
N_SAMPLES = 48_000 + (N_TOKENS - (48_000 % N_TOKENS)) % N_TOKENS
EMBED_DIM = 128
HIDDEN_DIM = 256
ENCODER_BLOCKS = 4
N_HEADS = 8
DROPOUT = 0.1
NOISE_SD = 0.2
N_CLASSES = 10

# training
ETA = 1e-6
L2_LAM = 0. ###! unimplemented
BATCH_SIZE = 64
N_EPOCHS = 100

# data
VAL_RATIO = 0.10
TEST_IDX = 1

# tracing
VERBOSE = True

assert (N_SAMPLES % N_TOKENS) == 0


### prepare data

# load data
data = reload_cache('data/urbansound8k_mono_24khz_float32.csv')

# expand sequences
data = expand_data(
	data.apply(expand_fn, **{'n_samples':N_SAMPLES}, axis=1)
)
plot_distributions(data.apply(wav_stats_fn, axis=1), filename='data/urbansound8k_description_t5.png')

# transform data
data = transform_data(
	data,
	[pad_and_slice_fn, cls_token_fn],
	[{'n_samples':N_SAMPLES, 'n_tokens':N_TOKENS}, {}],
	verbose=VERBOSE
)

# partition data
(train_x, train_y), (val_x, val_y), (test_x, test_y) = partition_data(
	data,
	test_idx=TEST_IDX,
	val_ratio=VAL_RATIO,
	verbose=VERBOSE
)

# plot sample
plot_tokenized_sample(train_x, prefix=f'{__file__.replace(".py","")}_input')

###! todo


# convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(buffer_size=len(train_x)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(BATCH_SIZE)

# memory cleanup
del data
del train_x; del val_x; del test_x
del train_y; del val_y; del test_y

# trace
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### initialise model

loss_fn = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.AdamW(learning_rate=ETA)

model = get_denoising_transformer_encoder(
	K2,
	N_TOKENS,
	N_SAMPLES//N_TOKENS,
	EMBED_DIM,
	HIDDEN_DIM,
	ENCODER_BLOCKS,
	n_heads=N_HEADS,
	dropout=DROPOUT,
	noise_sd=NOISE_SD,
)
model = convert_dte_to_classifier(
	model, 
	N_CLASSES,
	name='transformer_encoder_classifier_from_random_init'
)
model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
model.summary()

checkpoint_callback = callbacks.ModelCheckpoint(
	filepath=f'{model.name}.weights.h5',
	save_weights_only=True,
	monitor='val_loss',
	mode='min',
	save_best_only=True
)

print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### train model

train_history = model.fit(
	train_dataset,
	epochs=N_EPOCHS,
	validation_data=val_dataset,
	callbacks=[checkpoint_callback],
	verbose=int(VERBOSE)
).history

print(train_history)
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### evaluate model

model.load_weights(f'{model.name}.weights.h5')
test_history = model.evaluate(
	test_dataset,
	verbose=int(VERBOSE),
	return_dict=True
)

print(test_history)
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### save history

with open(f'{model.name}.history.pkl', 'wb') as f:
	pickle.dump({
		'train':train_history,
		'test':test_history
	}, f)


### plot history

plt.title('Classification')
plt.plot(train_history['loss'], label='train')
plt.plot(train_history['val_loss'], label='val', c='r')
plt.scatter([len(train_history['loss'])-1], test_history['loss'], label='test', c='g', marker='x')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.savefig(f'{model.name}.history.png')
plt.close()
