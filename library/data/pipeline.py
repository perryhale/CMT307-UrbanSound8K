import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import math

# import multiprocessing
# from pandarallel import pandarallel; pandarallel.initialize(nb_workers=multiprocessing.cpu_count())


def mono_avg_fn(row):
	""" Convert dual+ to mono by channel averaging
	# type: (pd.Series) -> pd.Series
	"""
	
	# unpack
	data = row['data']
	
	# transform
	new_row = row.copy()
	new_row['data'] = np.mean(data, axis=1) if (len(data.shape) > 1) else data
	
	return new_row


def rescale_fn(row, high=+1., low=-1., epsilon=1e-9):
	""" Min-Max scale raw audio sequence
	# type: (pd.Series, float, float) -> pd.Series
	"""
	
	# unpack
	data = row['data']
	
	# transform
	new_row = row.copy()
	new_row['data'] = low+(high-low) * (data-np.min(data)) / (np.max(data)-np.min(data)+epsilon)
	
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


def expand_fn(row, n_samples=96256):
	""" Pad and slice raw audio sequence into at least one slice of size slice_dim
	# type: (pd.Series, int) -> pd.Series
	"""
	
	# unpack
	data = row['data']
	n_padded = math.ceil(data.shape[0] / n_samples) * n_samples
	n_slices = n_padded//n_samples
	
	# transform
	new_row = pad_and_slice_fn(row, n_samples=n_padded, n_tokens=n_slices)
	
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


def natural_noise_fn(row, data=None):
	""" Randomly overlay samples to produce natural noise
	
	Function stub:
	- Should use with expand_data
	- Implementation should be deterministic with randomness handled in data arg prior to call
	"""
	assert data is not None, "Must pass data kwarg."
	pass


def expand_data(data):
	""" Expand sliced audio data into new dataframe of slices
	# type: (pd.DataFrame) -> pd.DataFrame
	"""
	
	# populate list of rows
	new_rows = []
	for _, row in data.iterrows():
		for sequence in row['data']:
			new_row = row.copy().to_dict() # to drop indexing from copy
			new_row['data'] = sequence
			new_rows.append(new_row)
	
	# construct new dataframe
	data = pd.DataFrame(new_rows)
	
	return data


def transform_data(
		data,
		transforms,
		transform_kwargs=[],
		callbacks=[],
		verbose=True
	):
	""" Apply sequential transformations to dataframe rows
	# type: (
		pd.DataFrame,
		List[Callable[[pd.Series, ...], pd.Series]],
		List[Dict[str, ...]],
		List[Callable[[int, pd.Series], None]],
		bool
	) -> pd.DataFrame
	
	+ data: pandas DataFrame with minumum columns ['rate', 'data', 'class'] + ['fold'] + ...
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
		
		# define exception wrapper
		# type: (pd.Series) -> pd.Series
		def exception_wrapper(row):
			try:
				return transform(row, **kwargs)
			except Exception as e:
				if verbose:
					print(f'Error in transform {i+1} {transform}, skipping row.')
					print(f' -> Got argument {type(row)}: {row}')
					print(f' -> Got exception: {e}')
				row_copy = row.copy()
				row_copy[:] = np.nan
				return row_copy
		
		# map transformation over dataframe
		data = data.apply(exception_wrapper, axis=1).dropna()
		
		# run callback
		for callback in callbacks:
			callback(i, data)
	
	return data


def partition_data(
		data,
		test_idx=1,
		test_ratio=0.25,
		val_ratio=0.1,
		verbose=True
	):
	""" Partition data
	# type: (pd.DataFrame, int, float, float, bool) -> Tuple[Tuple[np.ndarray]]
	
	+ data: pandas DataFrame with columns ['rate', 'data', 'fold', 'class']
	+ test_idx: determines which data fold will be reserved for testing
	+ test_ratio: ratio of dataset to reserve for testing if no fold information is provided, does not shuffle rows
	+ val_ratio: determines what proportion of the traing dataset will be reserved for validation
	+ verbose: print basic statistics
	"""
	
	# handle no fold information
	if 'fold' not in data.keys():
		n_folds = int(1/test_ratio)
		#data['fold'] = pd.Series(np.random.randint(1, n_folds+1, data.shape[0]))
		data['fold'] = np.tile(range(1, n_folds+1), len(data)//n_folds+1)[:len(data)] # deterministic expression
	
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


def dataset_generator(data_x, data_y, batch_size, shuffle=True, debug_title=None):
	""" Dataset batch generator
	# type: (np.ndarray, np.ndarray, int, bool, str) ~> Tuple[np.ndarray, np.ndarray]
	
	Yields optionally shuffled batches an infinite number of times
	"""
	assert (len(data_x)==len(data_y))
	n_samples = len(data_x)
	while True:
		data_idx_shuffle = np.random.permutation(n_samples) if shuffle else range(n_samples)
		for batch_idx, data_idx in enumerate(range(0, n_samples, batch_size)):
			if debug_title is not None:
				print(f'\n{debug_title} {batch_idx}')
			batch_data_idx = data_idx_shuffle[data_idx:data_idx+batch_size]
			batch_x = data_x[batch_data_idx]
			batch_y = data_y[batch_data_idx]
			yield batch_x, batch_y


def dataset_signature(data_x, data_y):
	""" Dataset tensor specification
	# type: (np.ndarray, np.ndarray) -> Tuple[tf.TensorSpec, tf.TensorSpec]
	
	Describes shape and type signature for x and y sets
	"""
	sig = (
		tf.TensorSpec(shape=(None, *data_x.shape[1:]), dtype=data_x.dtype),
		tf.TensorSpec(shape=(None, *data_y.shape[1:]), dtype=data_y.dtype)
	)
	return sig
