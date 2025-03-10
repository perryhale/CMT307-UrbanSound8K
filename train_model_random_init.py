import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks

import time
import random
import pickle

from library.random import split_key
from library.data import reload_cache
from library.models import get_denoising_transformer_encoder, convert_dte_to_classifier


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
N_SAMPLES = 96_000 + (96_000 % N_TOKENS)
EMBED_DIM = 128
HIDDEN_DIM = 256
ENCODER_BLOCKS = 2
N_CLASSES = 10

# training
ETA = 1e-3
L2_LAM = 0. ###! unimplemented
BATCH_SIZE = 32
N_EPOCHS = 10

# data
VAL_RATIO = 0.10
TEST_IDX = 1

# tracing
VERBOSE = True
VERBOSE_LVL = 3

assert (N_SAMPLES % N_TOKENS) == 0


### prepare data

# load data
data = reload_cache('data/urbansound8k_mono_24khz_float32.csv')

# define transforms
transforms=[pad_and_slice_fn, cls_token_fn]
transform_kwargs=[{'n_samples':N_SAMPLES, 'n_tokens':N_TOKENS}, {}]

# transform and partition data
train_dataset, val_dataset, test_dataset = prepare_data(
	data,
	test_idx=TEST_IDX,
	val_ratio=VAL_RATIO,
	batch_size=BATCH_SIZE,
	transforms=transforms
	transform_kwargs=transform_kwargsm
	verbose=VERBOSE_LVL*int(VERBOSE)
)

# memory cleanup
del data
del transforms
del transform_kwargs


### initialise model

loss_fn = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.AdamW(learning_rate=ETA)

model = get_denoising_transformer_encoder(
	K2,
	N_TOKENS,
	N_SAMPLES//N_TOKENS,
	EMBED_DIM,
	HIDDEN_DIM,
	ENCODER_BLOCKS
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
