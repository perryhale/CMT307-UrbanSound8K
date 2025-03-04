import numpy as np
import pandas as pd
import scipy as sp
import base64

""" https://urbansounddataset.weebly.com/download-urbansound8k.html """


def get_urbansound8k(
		root_path,
		target_rate=24000,
		dtype='float32',
		truncation=None,
		verbose=False
	):
	""" Load the UrbanSound8K data files into memory
	# type: (str, int, int, bool) -> Tuple(pd.DataFrame, Dict[int:str], pd.DataFrame)
	
	File handling:
	0. load files, skip incompatible formats
	Applies preprocessing steps:
	1. scale to {-1..+1}
	2. convert to mono by channel averaging
	3. resample to target rate with fourier method """
	
	# define wav loader function with error handling
	# type: (str) -> pd.Series
	def load_fn(path):
		try:
			rate, data = sp.io.wavfile.read(path)
			return pd.Series({'rate':rate, 'data':data})
		except ValueError as e:
			if verbose:
				print(f'Error reading \"{path}\", skipping.')
				print(e)
			return pd.Series({'rate':None, 'data':None})
	
	# load and parse metadata
	metadata = pd.read_csv(f'{root_path}/metadata/UrbanSound8K.csv')
	metadata['path'] = metadata.apply(lambda row : f'{root_path}/audio/fold{row["fold"]}/{row["slice_file_name"]}', axis=1)
	class_names = {k:v for k,v in sorted({int(k):v for k,v in zip(metadata['classID'].unique(), metadata['class'].unique())}.items())}
	if truncation is not None:
		metadata = metadata[:truncation]
	if verbose:
		print(class_names)
	
	# load data
	data = pd.DataFrame() # [rate, data, fold, class]
	data[['rate', 'data']] = metadata['path'].apply(load_fn)
	data['fold'] = metadata['fold']
	data['class'] = metadata['classID']
	data = data.dropna()
	if verbose:
		print(data)
	
	# scale to -1..+1
	if verbose:
		print(data['data'][0].dtype)
		print(data['data'].apply(lambda x : np.min(x).item()).min())
		print(data['data'].apply(lambda x : np.max(x).item()).max())
	data['data'] = data['data'].apply(lambda x : x.astype(dtype))
	data['data'] = data['data'].apply(lambda x : -1+2*(x-np.min(x)) / (np.max(x)-np.min(x)))
	if verbose:
		print(data['data'][0].dtype)
		print(data['data'].apply(lambda x : np.min(x).item()).min())
		print(data['data'].apply(lambda x : np.max(x).item()).max())
		print(data)
	
	# convert dual to mono by channel averaging
	data['data'] = data['data'].apply(lambda x : np.mean(x, axis=1) if x.shape[-1]==2 else x)
	if verbose:
		print(data)
	
	# resample at target_rate Hz
	data[['rate', 'data']] = data.apply(lambda row : pd.Series({
		'rate' : target_rate,
		'data': sp.signal.resample(row['data'], round(len(row['data']) * float(target_rate) / row['rate']))
	}) if target_rate != int(row['rate']) else pd.Series({
		'rate' : row['rate'],
		'data' : row['data']
	}), axis=1)
	if verbose:
		print(data)
	
	return metadata, class_names, data


def reload_cache(path, cache_dtype='INFER'):
	""" Load cached 'data' csv file into memory
	# type: (str, str) -> pd.DataFrame
	
	Infers data type from file name by default """
	
	# infer dtype
	if cache_dtype == 'INFER':
		cache_dtype = path.replace('.csv','').split('_')[-1]
	
	# load
	data = pd.read_csv(path)
	data['data'] = data['data'].apply(lambda x : np.frombuffer(base64.b64decode(x), dtype=cache_dtype))
	
	# force rescale
	data['data'] = data['data'].apply(lambda x : -1+2*(x-np.min(x)) / (np.max(x)-np.min(x)))
	
	return data


def create_cache(
		root_path,
		out_path='.',
		target_rate=24000,
		cache_dtype='float64',
		verbose=False
	):
	""" Create preprocessed UrbanSound8K data cache
	# type: (str, str, float, str, bool, bool) -> bool
	
	"""
	
	# load and preprocess
	metadata, class_names, data = get_urbansound8k(
		root_path,
		target_rate=target_rate,
		dtype=cache_dtype,
		verbose=verbose
	)
	
	try:
		
		# cache class names
		class_names_df = pd.DataFrame.from_dict(class_names, orient='index')
		class_names_df.to_csv(f'{out_path}/urbansound8k_classes.csv', header=['name'], index_label='id')
		
		# cache data
		data['data'] = data['data'].apply(lambda x : base64.b64encode(x.astype(cache_dtype).tobytes()).decode('utf-8'))
		data.to_csv(f'{out_path}/urbansound8k_mono_{int(target_rate/1000)}khz_{cache_dtype}.csv', index=False)
		
		return True
	
	except Exception as e:
		print('Error writing cache:')
		print(e)
		
		return False


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
