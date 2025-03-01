import numpy as np
import time
import pickle
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
K2 = split_key(K1)[1]


### hyperparameters

# architecture
N_SAMPLES = 96_000
N_TOKENS = 256
EMBED_DIM = 256
HIDDEN_DIM = 256
ENCODER_BLOCKS = 1

# training
ETA = 1e-6
L2_LAM = 0. ###! unimplemented
BATCH_SIZE = 32
N_EPOCHS = 30
VAL_RATIO = 0.10

# tracing
VERBOSE = True

assert (N_SAMPLES % N_TOKENS) == 0


### prepare data

# load data
data = reload_cache('data/urbansound8k_mono_24khz_float32.csv')

# pad and tokenize sequences
data['data'] = data['data'].apply(lambda x : np.array(np.split(np.pad(x, (0,N_SAMPLES-len(x))) if len(x) < N_SAMPLES else x[:N_SAMPLES], N_TOKENS)))
print(data)

# partition data
train_x = np.array(list(data[(data['fold'] != 1)]['data']))
test_x = np.array(list(data[(data['fold'] == 1)]['data']))

# trace
print(train_x.shape, 'train')
print(test_x.shape, 'test')
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### initialise model

loss_fn = losses.MeanSquaredError()
optimizer = optimizers.AdamW(learning_rate=ETA)

model = get_denoising_transformer_encoder(
	K1, 
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


### train model

train_history = model.fit(
	train_x,
	train_x,
	epochs=N_EPOCHS,
	validation_split=VAL_RATIO,
	batch_size=BATCH_SIZE,
	callbacks=[checkpoint_callback],
	verbose=int(VERBOSE)
).history

print(train_history)

with open('train_model_unsupervised_history.pkl', 'wb') as f:
	pickle.dump(train_history, f)
