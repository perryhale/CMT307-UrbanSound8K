import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks

from library.random import split_key
from library.data import reload_cache
from library.models import get_denoising_transformer_encoder


### setup

# start timer
T0 = time.time()
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# init RNG seeds
K0 = 999
K1 = split_key(K0)[1]


### hyperparameters

# architecture
N_TOKENS = 512
N_SAMPLES = 96_000 + (96_000 % N_TOKENS)
EMBED_DIM = 128
HIDDEN_DIM = 256
ENCODER_BLOCKS = 2

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

assert (N_SAMPLES % N_TOKENS) == 0


### prepare data

# load data
data = reload_cache('data/urbansound8k_mono_24khz_float32.csv')

# pad and slice sequences
data['data'] = data['data'].apply(lambda x : np.array(np.split(np.pad(x, (0, N_SAMPLES-len(x))) if len(x) < N_SAMPLES else x[:N_SAMPLES], N_TOKENS)))
print(data)

# partition data
train_idx = (data['fold'] != TEST_IDX)
test_idx = (data['fold'] == TEST_IDX)
train_x = np.array(list(data[train_idx]['data']))
val_x = train_x[int(len(train_x)*(1-VAL_RATIO)):]
train_x = train_x[:int(len(train_x)*(1-VAL_RATIO))]
test_x = np.array(list(data[test_idx]['data']))

# plot sample
plt.imshow(train_x[np.random.randint(0, len(train_x)-1)])
plt.show()

# trace
print(train_x.shape, 'train')
print(test_x.shape, 'test')
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_x)).shuffle(buffer_size=len(train_x)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_x)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_x)).batch(BATCH_SIZE)

# memory cleanup
del data
del train_idx; del test_idx
del train_x
del val_x
del test_x


### initialise model

loss_fn = losses.MeanSquaredError()
optimizer = optimizers.AdamW(learning_rate=ETA)

model = get_denoising_transformer_encoder(
	K1,
	N_TOKENS,
	N_SAMPLES//N_TOKENS,
	EMBED_DIM,
	HIDDEN_DIM,
	ENCODER_BLOCKS
)
model.compile(loss=loss_fn, optimizer=optimizer, metrics=['root_mean_squared_error'])
model.summary()

checkpoint_callback = callbacks.ModelCheckpoint(
	filepath=f'{model.name}.weights.h5'.replace('-','_').lower(),
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

model.load_weights(f'{model.name}.weights.h5'.replace('-','_').lower())
test_history = model.evaluate(
	test_dataset,
	verbose=int(VERBOSE),
	return_dict=True
)

print(test_history)
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### save history

with open(f'{model.name}_history.pkl'.replace('-','_').lower(), 'wb') as f:
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
plt.show()
