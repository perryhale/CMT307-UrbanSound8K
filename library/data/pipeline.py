import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import abc


def mono_avg_fn(row):
	""" Convert dual to mono by channel averaging
	# type: (pd.Series) -> pd.Series
	
	Preserves adjacent columns
	"""
	
	# unpack
	data = row['data']
	
	# transform
	new_row = row.copy()
	new_row['data'] = np.mean(data, axis=1) if data.shape[-1] == 2 else data
	
	return new_row


def rescale_fn(row, high=+1., low=-1.):
	""" Min-Max scale raw audio sequence
	# type: (pd.Series, float, float) -> pd.Series
	
	Preserves adjacent columns
	"""
	
	# unpack
	data = row['data']
	
	# transform
	new_row = row.copy()
	new_row['data'] = low+(high-low) * (data-np.min(data)) / (np.max(data)-np.min(data))
	
	return new_row


def resample_fn(row, target_rate=24_000):
	""" Resample entries in data column to target_rate using fourier method
	# type: (pd.Series, int) -> pd.Series
	
	Preserves adjacent columns
	"""
	
	# unpack
	rate = row['rate']
	data = row['data']
	
	# transform
	new_row = row.copy()
	if int(target_rate) != int(rate):
		new_row['rate'] = target_rate
		new_row['data'] = sp.signal.resample(data, round(len(data) * (float(target_rate) / rate)))
	
	return new_row


def pad_and_slice_fn(row, n_samples=96256, n_tokens=512):
	""" Pad and slice raw audio sequences into raw tokens
	# type: (pd.Series, int, int) -> pd.Series
	
	Preserves adjacent columns
	
	Default args produce 3.90625ms tokens @24KHz
	n_samples must be divisible by n_tokens
	"""
	assert (n_samples % n_tokens) == 0
	
	# unpack
	data = row['data']
	
	# transform
	new_row = row.copy()
	new_row['data'] = np.array(np.split(np.pad(data, (0, n_samples - len(data))) if len(data) < n_samples else data[:n_samples], n_tokens))
	
	return new_row


def cls_token_fn(row):
	""" Insert cls token at first index of tokenized sequence
	# type: (pd.Series) -> pd.Series
	
	Preserves adjacent columns
	"""
	
	# unpack
	data = row['data']
	
	# transform
	new_row = row.copy()
	new_row['data'] = np.concatenate((np.array([[np.sign((i % 2) - 0.5) for i in range(data.shape[1])]]), data[1:, :]))
	
	return new_row


def transform_pipeline(
		data,
		transforms,
		transform_kwargs=[{}]
		verbose=False
	):
	
	# apply data transformations
	for transform, kwargs in zip(transforms, transform_kwargs):
		data = data.apply(transform, **kwargs, axis=1)
		if verbose:
			print(data)
	
	return data




def prepare_data(
		data,
		test_idx=1,
		val_ratio=0.1,
		batch_size=64,
		transforms=[pad_and_slice_fn], # assumes pre-processed
		transform_kwargs=[{}], # uses default args
		plot_key=None,
		plot_title='input_sample',
		verbose=1
	):
	""" Prepare data with transformation pipeline and partitioning
	# type: (pd.DataFrame, int, float, int, List[Callable[[pd.Series, ...], pd.Series]], List[Dict[str, ...]], int) -> Tuple[Tuple[np.ndarray]]
	
	+ data: pandas DataFrame with columns ['rate', 'data', 'fold', 'class']
	+ test_idx: determines which data fold will be reserved for testing
	+ val_ratio: determines what proportion of the traing dataset will be reserved for validation
	+ batch_size: determines the size of the training batches
	+ transforms: list of preprocessing transforms, applied sequentially to data rows
	+ transform_kwargs: list of keyword arguments for each transform
	+ verbose: enumerable argument
		0: silent tracing
		1: print basic statistics
		2: print intermediate samples during transform
		3: random sample plotting with plt
	+ plot_title: prefix for plot filename if verbose>=3
	"""
	
	# apply data transformations
	for transform, kwargs in zip(transforms, transform_kwargs):
		data = data.apply(transform, **kwargs, axis=1)
		if verbose > 1:
			print(data)
	
	# partition data
	train_idx = (data['fold'] != test_idx)
	test_idx = (data['fold'] == test_idx)
	train_x = np.array(list(data[train_idx]['data']))
	train_y = np.array(list(data[train_idx]['class']))[:, np.newaxis]
	val_x = train_x[int(len(train_x)*(1-val_ratio)):]
	val_y = train_y[int(len(train_y)*(1-val_ratio)):]
	train_x = train_x[:int(len(train_x)*(1-val_ratio))]
	train_y = train_y[:int(len(train_y)*(1-val_ratio))]
	test_x = np.array(list(data[test_idx]['data']))
	test_y = np.array(list(data[test_idx]['class']))[:, np.newaxis]
	
	# handle plotting kwargs
	plot_key = (plot_kwargs['key'] if 'key' in plot_kwargs.keys() else None)
	plot_title = (plot_kwargs['title'] if 'title' in plot_kwargs.keys() else 'sample_input')
	
	# plot sample
	if verbose > 2:
		x_sample = train_x[np.random.default_rng(seed=).integers(0, len(train_x)-1)]
		plt.figure(figsize=(4,10))
		plt.imshow(x_sample)
		plt.savefig(f'{plot_title}-001.png')
		plt.close()
		plt.figure(figsize=(10,3))
		plt.imshow(x_sample[:16])
		plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
		plt.savefig(f'{plot_title}-002.png')
		plt.subplots_adjust()
		plt.close()
	
	# trace
	if verbose > 0: 
		print(train_x.shape, train_y.shape, 'train')
		print(val_x.shape, val_y.shape, 'val')
		print(test_x.shape, test_y.shape, 'test')
	
	return (train_x, train_y), (val_x, val_y), (test_x, test_y)
