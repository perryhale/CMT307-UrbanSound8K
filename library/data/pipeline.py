import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import abc

""" Preprocessing functions, applied over rows of dataframe
"""

def mono_avg_fn(row):
	""" Convert dual to mono by channel averaging
	# type: (pd.Series) -> pd.Series
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
	
	Default args produce 7.8125ms tokens for 4 seconds at @24KHz
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
	"""
	
	# unpack
	data = row['data']
	
	# transform
	new_row = row.copy()
	new_row['data'] = np.concatenate((np.array([[np.sign((i % 2) - 0.5) for i in range(data.shape[1])]]), data[1:, :]))
	
	return new_row


def transform_data(
		data,
		transforms,
		transform_kwargs=[],
		callbacks=[],
		verbose=False
	):
	""" Apply sequential transformations to dataframe rows
	# type: (
		pd.DataFrame, 
		List[Callable[[pd.Series, ...], pd.Series]], 
		List[Dict[str, ...]], 
		List[Callable[[int, pd.Series], None]], 
		bool
	) -> pd.DataFrame
	
	+ data: pandas DataFrame with columns ['rate', 'data', 'fold', 'class']
	+ transforms: list of preprocessing transforms. assmues (pd.Series, **) -> pd.Series applied to data rows
	+ transform_kwargs: list of keyword arguments for each transform
	+ callbacks: functions called every transform on data, each takes int argmuent representing transform index and current data object
	"""
	
	# handle kwargs
	if transform_kwargs is []:
		transform_kwargs = [{} for _ in range(len(transforms))]
	else:
		assert len(transforms)==len(transform_kwargs), "Num transforms must equal num kwargs."
	
	# apply transform sequence
	for i, (transform, kwargs) in enumerate(zip(transforms, transform_kwargs)):
		data = data.apply(transform, **kwargs, axis=1)
		for callback in callbacks:
			callback(i, data)
		if verbose:
			print(data)
	
	return data


def partition_data(
		data,
		test_idx=1,
		val_ratio=0.1,
		batch_size=64,
		verbose=True
	):
	""" Partition data
	# type: (pd.DataFrame, int, float, int, bool) -> Tuple[Tuple[np.ndarray]]
	
	+ data: pandas DataFrame with columns ['rate', 'data', 'fold', 'class']
	+ test_idx: determines which data fold will be reserved for testing
	+ val_ratio: determines what proportion of the traing dataset will be reserved for validation
	+ batch_size: determines the size of the training batches
	+ verbose: print basic statistics
	"""
	
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
	
	# trace
	if verbose: 
		print(train_x.shape, train_y.shape, 'train')
		print(val_x.shape, val_y.shape, 'val')
		print(test_x.shape, test_y.shape, 'test')
	
	return (train_x, train_y), (val_x, val_y), (test_x, test_y)
