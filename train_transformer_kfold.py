import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	precision_recall_fscore_support
)

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
from library.data.descriptive import (
	wav_stats_fn,
	plot_distributions,
	plot_tokenized_sample
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
PATIENCE = 10

# learning rate
ETA = 1e-3
DECAY_RATE = 0.37
DECAY_FACTOR = 0.1

# explicit regularization
L2_LAM = 0. ###! unimplemented
DROPOUT = 0.4

# data partitions
VAL_RATIO = 0.10

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


### train kfold

fold_idx = np.unique(data['fold'])
history = np.empty(len(fold_idx), dtype='object')

for test_idx in fold_idx:
	
	
	### partition data
	
	(train_x, train_y), (val_x, val_y), (test_x, test_y) = partition_data(
		data,
		test_idx=test_idx,
		val_ratio=VAL_RATIO,
		verbose=VERBOSE
	)
	
	train_dataset = tf.data.Dataset.from_generator(
		lambda:batch_generator(train_x, train_y, BATCH_SIZE, shuffle=True),
		output_signature=batch_signature(train_x, train_y)
	).prefetch(tf.data.experimental.AUTOTUNE)

	val_dataset = tf.data.Dataset.from_generator(
		lambda:batch_generator(val_x, val_y, BATCH_SIZE, shuffle=False),
		output_signature=batch_signature(val_x, val_y)
	).prefetch(tf.data.experimental.AUTOTUNE)
	
	test_iterator = batch_generator(test_x, test_y, BATCH_SIZE, shuffle=False)
	
	train_steps = len(train_x)//BATCH_SIZE
	val_steps = len(val_x)//BATCH_SIZE
	test_steps = len(test_x)//BATCH_SIZE
	
	
	### initialise model
	
	loss_fn = losses.SparseCategoricalCrossentropy()
	lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
		initial_learning_rate=ETA,
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
		dropout=DROPOUT
	)
	model.load_weights('denoising_transformer_encoder.weights.h5')
	model = convert_dte_to_classifier(
		model,
		N_CLASSES,
		name=f'transformer_encoder_classifier_from_dte_backbone_fold{test_idx}'
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
	
	# reload weights
	model.load_weights(f'{model.name}.weights.h5')
	
	# compute predictions
	test_y = []
	test_yh = []
	for _ in range(test_steps):
		batch_x, batch_y = next(test_iterator)
		batch_yh = model.predict(batch_x, verbose=0)
		test_y.append(batch_y)
		test_yh.append(batch_yh)
	
	test_y = np.concatenate(test_y, axis=0)
	test_yh = np.concatenate(test_yh, axis=0)
	
	# compute scores
	test_loss = float(model.loss(test_y, test_yh).numpy())
	test_accuracy = float(accuracy_score(test_y, np.argmax(test_yh, axis=1)))
	test_cfm = confusion_matrix(test_y, np.argmax(test_yh, axis=1), labels=range(10))
	test_prfs = precision_recall_fscore_support(test_y, np.argmax(test_yh, axis=1), labels=range(10), average='weighted', zero_division=0.0)
	test_precision, test_recall, test_f1 = [float(x) for x in test_prfs[:-1]]
	
	# record results
	test_history = {
		'loss':test_loss,
		'accuracy':test_accuracy,
		'confusion':test_cfm,
		'precision':test_precision,
		'recall':test_recall,
		'f1':test_f1
	}
	
	print(test_history)
	print(f'[Elapsed time: {time.time()-T0:.2f}s]')
	
	
	### record results
	
	history[test_idx-1] = {
		'info':{
			'name':model.name,
			'fold':fold_idx
		},
		'train':train_history,
		'test':test_history
	}
	
	with open(f'{__file__.replace(".py", "")}.history.pkl', 'wb') as f:
		pickle.dump(history, f)
