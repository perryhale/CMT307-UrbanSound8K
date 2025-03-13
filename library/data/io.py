import numpy as np
import pandas as pd
import scipy as sp
import base64
from .pipeline import rescale_fn


def load_wav_fn(path, verbose=True):
	""" Load a wav file from disk
	# type: (str, bool) -> pd.Series
	"""
	try:
		rate, data = sp.io.wavfile.read(path)
		return pd.Series({'rate': rate, 'data': data})
	except ValueError as e:
		if verbose:
			print(f'Error reading \"{path}\", skipping.')
			print(e)
		return pd.Series({'rate': None, 'data': None})


def get_urbansound8k(
		root_path,
		dtype='float64',
		truncation=None,
		verbose=True
	):
	""" Load the UrbanSound8K data files into memory
	https://urbansounddataset.weebly.com/download-urbansound8k.html
	# type: (str, str, int, bool) -> Tuple(pd.DataFrame, Dict[int:str], pd.DataFrame)
	
	* skips incompatible formats
	* converts to homogenous dtype
	* if truncated, data is first shuffled
	"""
	
	# load metadata
	metadata = pd.read_csv(f'{root_path}/metadata/UrbanSound8K.csv')
	metadata['path'] = metadata.apply(lambda row : f'{root_path}/audio/fold{row["fold"]}/{row["slice_file_name"]}', axis=1)
	
	# truncate data (opt)
	if truncation is not None:
		metadata = metadata.sample(frac=1)
		metadata = metadata[:truncation]
	
	# determine class names
	class_names = {k:v for k,v in sorted({int(k):v for k,v in zip(metadata['classID'].unique(), metadata['class'].unique())}.items())}
	
	# populate dataframe
	data = pd.DataFrame() # [rate, data, fold, class]
	data['fold'] = metadata['fold']
	data['class'] = metadata['classID']
	data[['rate', 'data']] = metadata['path'].apply(load_wav_fn, args=(verbose,))
	
	# drop and cast
	data = data.dropna()
	data['data'] = data['data'].apply(lambda x : x.astype(dtype))
	
	return metadata, class_names, data


def get_audioset(
		root_path,
		dtype='float64',
		truncation=None,
		verbose=True
	):
	""" Load the AudioSet data files into memory
	https://www.kaggle.com/datasets/zfturbo/audioset
	# type: (str, str, int, bool) -> Tuple(pd.DataFrame, Dict[int:str], pd.DataFrame)
	
	* skips incompatible formats
	* converts to homogenous dtype
	* if truncated, data is first shuffled
	"""
	
	# load metadata
	metadata = pd.read_csv(f'{root_path}/train.csv')
	metadata['path'] = metadata.apply(lambda row : f'{root_path}/train_wav/{row["YTID"]}.wav', axis=1)
	
	# truncate data (opt)
	if truncation is not None:
		metadata = metadata.sample(frac=1)
		metadata = metadata[:truncation]
	
	# determine class names
	class_metadata = pd.read_csv(f'{root_path}/class_labels_indices.csv')
	class_mid_to_idx = dict(zip(class_metadata['mid'], class_metadata['index']))
	class_names = dict(zip(class_metadata['index'], class_metadata['display_name']))
	
	# populate dataframe
	data = pd.DataFrame() # [rate, data, class]
	data[['rate', 'data']] = metadata['path'].apply(load_wav_fn, args=(verbose,))
	data['class'] = metadata['positive_labels'].apply(lambda x : [class_mid_to_idx[mid] for mid in x.split(',')]) # class_mid_to_idx.get(mid, None)
	
	# drop and cast
	data = data.dropna()
	data['data'] = data['data'].apply(lambda x : x.astype(dtype))
	
	return metadata, class_names, data


def create_cache(
		data,
		class_names=None,
		cache_dtype='float32',
		cache_root='.',
		cache_name='unnamed_cache'
	):
	""" Write data to cache
	# type: (pd.DataFrame, Dict[int:str], str, str, str) -> str
	"""
	
	# determine cache location
	cache_location = f'{cache_root}/{cache_name}_{cache_dtype}.csv'
	
	try:
		# cache class names
		if class_names is not None:
			class_names_df = pd.DataFrame.from_dict(class_names, orient='index')
			class_names_df.to_csv(cache_location.replace('.csv', '_classes.csv'), header=['name'], index_label='id')
		
		# cache data
		data['data'] = data['data'].apply(lambda x : base64.b64encode(x.astype(cache_dtype).tobytes()).decode('utf-8'))
		data.to_csv(cache_location, index=False)
		
		return cache_location
	
	except Exception as e:
		print('Error writing cache:')
		raise


def reload_cache(path, cache_dtype='INFER', rescale=True, rescale_kwargs={}):
	""" Read data from cache into memory
	# type: (str, str, bool, Dict[str:any]) -> pd.DataFrame
	
	Infers data type from file name by default
	"""
	
	try:
		# infer dtype
		if cache_dtype == 'INFER':
			cache_dtype = path.replace('.csv','').split('_')[-1]
		
		# load
		data = pd.read_csv(path)
		data['data'] = data['data'].apply(lambda x : np.frombuffer(base64.b64decode(x), dtype=cache_dtype))
		
		# rescale (opt)
		if rescale:
			data = data.apply(rescale_fn, **rescale_kwargs, axis=1)
		
		return data
	
	except Exception as e:
		print('Error reading cache:')
		raise


# # low pass filter + decimate undersampling
# from scipy.signal import butter, filtfilt, decimate
# def lowpass_filter(x, original_rate, target_rate):
	# nyquist_rate = target_rate / 2.0
	# cutoff_freq = nyquist_rate / (original_rate / 2.0)
	
	# print(cutoff_freq)
	
	# b, a = butter(4, cutoff_freq, btype='low', analog=False)
	# xhat = filtfilt(b, a, x)
	# return xhat
# xhat = decimate(lowpass_filter(x, original_rate, target_rate), original_rate // target_rate)
# # split pandas by fold
# data_folds = [data[data['fold']==fold_id] for fold_id in np.sort(data['fold'].unique())]
# return metadata, class_names, data_folds
