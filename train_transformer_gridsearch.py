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
	batch_generator,
	batch_signature
)
from library.models.transformer import (
	get_denoising_transformer_encoder,
	convert_dte_to_classifier
)


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
N_CLASSES = 10

assert (N_SAMPLES % N_TOKENS) == 0

# steps
N_EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 5

# learning rate
DECAY_RATE = 0.37
DECAY_FACTOR = 0.1

# data partitions
VAL_RATIO = 0.10
TEST_IDX = 1

# gridsearch
GS_ETA_HIGH = 1e-2
GS_ETA_LOW = 1e-6
GS_ETA_DOF = 8
GS_DROPOUT_HIGH = 0.5
GS_DROPOUT_LOW = 0.0
GS_DROPOUT_DOF = 8

# tracing
VERBOSE = True


### prepare data

# load data
print('Load cache..')
data = reload_cache('data/urbansound8k_mono_24khz_float32.csv')
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# expand sequences
print('Expand sequences..')
data = expand_data(data.apply(expand_fn, **{'n_samples':N_SAMPLES}, axis=1))
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# transform data
print('Apply transforms..')
data = transform_data(
	data,
	[pad_and_slice_fn, cls_token_fn],
	[{'n_samples':N_SAMPLES, 'n_tokens':N_TOKENS}, {}],
	verbose=VERBOSE
)
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# partition data
print('Partition data..')
(train_x, train_y), (val_x, val_y), (test_x, test_y) = partition_data(
	data,
	test_idx=TEST_IDX,
	val_ratio=VAL_RATIO,
	verbose=VERBOSE
)
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# convert to tf.data.Dataset
print('Convert to Dataset..')

train_dataset = tf.data.Dataset.from_generator(
	lambda:batch_generator(train_x, train_y, BATCH_SIZE, shuffle=True),#, debug_title='train_dataset'),
	output_signature=batch_signature(train_x, train_y)
).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
	lambda:batch_generator(val_x, val_y, BATCH_SIZE, shuffle=False),#, debug_title='val_dataset'),
	output_signature=batch_signature(val_x, val_y)
).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
	lambda:batch_generator(test_x, test_y, BATCH_SIZE, shuffle=False),#, debug_title='test_dataset'),
	output_signature=batch_signature(test_x, test_y)
).prefetch(tf.data.experimental.AUTOTUNE)

train_steps = len(train_x)//BATCH_SIZE
val_steps = len(val_x)//BATCH_SIZE
test_steps = len(test_x)//BATCH_SIZE

print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# memory cleanup
del data


### perform gridsearch

eta_space = np.linspace(GS_ETA_LOW, GS_ETA_HIGH, num=GS_ETA_DOF)
dropout_space = np.linspace(GS_DROPOUT_LOW, GS_DROPOUT_HIGH, num=GS_DROPOUT_DOF)
history = np.empty((len(eta_space), len(dropout_space)), dtype='object')

for i, eta in enumerate(eta_space):
	for j, dropout in enumerate(dropout_space):
		
		
		### initialise model
		
		loss_fn = losses.SparseCategoricalCrossentropy()
		lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
			initial_learning_rate=eta,
			decay_rate=DECAY_RATE,
			decay_steps=DECAY_FACTOR*N_EPOCHS*train_steps
		)
		optimizer = optimizers.AdamW(learning_rate=lr_schedule)
		
		model = get_denoising_transformer_encoder(
			K2,
			N_TOKENS,
			N_SAMPLES//N_TOKENS,
			EMBED_DIM,
			HIDDEN_DIM,
			ENCODER_BLOCKS,
			n_heads=N_HEADS,
			dropout=dropout
		)
		model.load_weights('denoising_transformer_encoder.weights.h5')
		model = convert_dte_to_classifier(
			model,
			N_CLASSES,
			name=f'{model.name}_eta{eta:.7f}_dropout{dropout:.7f}'.replace('.','_')
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
		early_stopping_callback = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
		
		print(f'[Elapsed time: {time.time()-T0:.2f}s]')
		
		
		### train model
		
		train_history = model.fit(
			train_dataset,
			epochs=N_EPOCHS,
			steps_per_epoch=train_steps,
			validation_data=val_dataset,
			validation_steps=val_steps,
			callbacks=[checkpoint_callback, early_stopping_callback],
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
		
		
		### record results
		
		history[i,j] = {
			'info':{
				'name':model.name,
				'eta':eta,
				'dropout':dropout
			},
			'train':train_history,
			'test':test_history
		}
		
		with open(f'{__file__.replace(".py", "")}.history.pkl', 'wb') as f:
			pickle.dump(history, f)
