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
	expand_fn,
	expand_data,
	transform_data,
	partition_data
)
from library.data.descriptive import (
	wav_stats_fn,
	plot_distributions,
	plot_tokenized_sample
)
from library.models.transformer import get_denoising_transformer_encoder


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

# training
ETA = 1e-5
L2_LAM = 0. ###! unimplemented
BATCH_SIZE = 32
N_EPOCHS = 50

# data
VAL_RATIO = 0.10
TEST_RATIO = 0.25

# tracing
VERBOSE = True

assert (N_SAMPLES % N_TOKENS) == 0


### prepare data

# # load data
# print('Load cache..')
# data = reload_cache('data/audioset_mono_24khz_float32.csv')
# print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# # expand sequences
# print('Expand sequences..')
# data = expand_data(
	# data.apply(expand_fn, **{'n_samples':N_SAMPLES}, axis=1)
# )
# plot_distributions(data.apply(wav_stats_fn, axis=1), filename='data/audioset_description_t5.png')
# print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# # transform data
# print('Apply transforms..')
# data = transform_data(
	# data,
	# [pad_and_slice_fn],
	# [{'n_samples':N_SAMPLES, 'n_tokens':N_TOKENS}],
	# verbose=VERBOSE
# )
# print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# # partition data
# print('Partition data..')
# (train_x, _), (val_x, _), (test_x, _) = partition_data(
	# data,
	# test_ratio=TEST_RATIO,
	# val_ratio=VAL_RATIO,
	# verbose=VERBOSE
# )



###! debug
# with open('dataset_debug_cache.pkl', 'wb') as f:
	# partitions = partition_data(
		# data,
		# test_ratio=TEST_RATIO,
		# val_ratio=VAL_RATIO,
		# verbose=VERBOSE
	# )
	# pickle.dump(partitions, f)
with open('dataset_debug_cache.pkl', 'rb') as f:
	(train_x, _), (val_x, _), (test_x, _) = pickle.load(f)
	print(train_x.shape, train_x.shape, 'train')
	print(val_x.shape, val_x.shape, 'val')
	print(test_x.shape, test_x.shape, 'test')



plot_tokenized_sample(train_x, prefix=f'{__file__.replace(".py","")}_input')
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# convert to tf.data.Dataset
print('Convert to Dataset..')

def dataset_gen(data_x, data_y, batch_size, shuffle=True, debug_title=None):
	assert (len(data_x)==len(data_y))
	n_samples = len(data_x)
	while True:
		shuffle_idx = np.random.permutation(n_samples) if shuffle else range(n_samples)
		for debug_i,i in enumerate(range(0, n_samples, batch_size)):
			if debug_title is not None:
				print(f'\n{debug_title} {debug_i}')
			batch_idx = shuffle_idx[i:i+batch_size]
			batch_x = data_x[batch_idx]
			batch_y = data_y[batch_idx]
			yield batch_x, batch_y

def dataset_sig(data_x, data_y):
	sig = (
		tf.TensorSpec(shape=(None, *data_x.shape[1:]), dtype=data_x.dtype),
		tf.TensorSpec(shape=(None, *data_y.shape[1:]), dtype=data_y.dtype)
	)
	return sig

train_dataset = tf.data.Dataset.from_generator(
	lambda: dataset_gen(train_x, train_x, BATCH_SIZE, shuffle=True, debug_title='train_dataset'),
	output_signature=dataset_sig(train_x, train_x)
).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
	lambda: dataset_gen(val_x, val_x, BATCH_SIZE, shuffle=False, debug_title='val_dataset'),
	output_signature=dataset_sig(val_x, val_x)
).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
	lambda: dataset_gen(test_x, test_x, BATCH_SIZE, shuffle=False, debug_title='test_dataset'),
	output_signature=dataset_sig(test_x, test_x)
).prefetch(tf.data.experimental.AUTOTUNE)

train_steps = len(train_x)//BATCH_SIZE
val_steps = len(val_x)//BATCH_SIZE
test_steps = len(test_x)//BATCH_SIZE

print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### initialise model

loss_fn = losses.MeanSquaredError()
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
model.compile(loss=loss_fn, optimizer=optimizer, metrics=['root_mean_squared_error'])
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
	steps_per_epoch=train_steps,
	validation_data=val_dataset,
	validation_steps=val_steps,
	callbacks=[checkpoint_callback],
	verbose=int(VERBOSE)
).history

print(train_history)
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### evaluate model

model.load_weights(f'{model.name}.weights.h5')
test_history = model.evaluate(
	test_dataset,
	steps=test_steps,
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

plt.title('Unsupervised Denoising')
plt.plot(train_history['loss'], label='train')
plt.plot(train_history['val_loss'], label='val', c='r')
plt.scatter([len(train_history['loss'])-1], test_history['loss'], label='test', c='g', marker='x')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.savefig(f'{model.name}.history.png')
plt.close()
