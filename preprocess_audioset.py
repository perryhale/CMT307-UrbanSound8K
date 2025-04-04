import scipy

from library.data.io import get_audioset, create_cache, reload_cache
from library.data.pipeline import (
	rescale_fn,
	mono_avg_fn,
	resample_fn,
	transform_data
)
from library.data.descriptive import wav_stats_fn, plot_distributions

""" https://www.kaggle.com/datasets/zfturbo/audioset """
""" Create AudioSet cache with debug and description actions"""


### arguments

ROOT_PATH = 'data/AudioSet'
OUTPUT_PATH = 'data/'
TARGET_RATE = 24000
CACHE_DTYPE = 'float32'
VERBOSE = True

DEBUG = True
###! max 19,644 rows
TRUNC = 1.
EG_IDX = 5


### callbacks

def debug_callback(i, df):
	if DEBUG:
		rate = int(df.iloc[EG_IDX,:]['rate'])
		sample = df.iloc[EG_IDX,:]['data']
		print(df)
		print(f'[{EG_IDX}]', sample.dtype, sample.shape)
		scipy.io.wavfile.write(f'{OUTPUT_PATH}/audioset_sample{EG_IDX}_t{i+1}.wav', rate, sample)

descriptive_callback = lambda i,df : plot_distributions(df.apply(wav_stats_fn, axis=1), filename=f'{OUTPUT_PATH}/audioset_description_t{i+1}.png')


### transforms

transforms = [rescale_fn, mono_avg_fn, resample_fn]
transform_kwargs = [{}, {}, {'target_rate' : TARGET_RATE}]


### main

# load data
metadata, class_names, data = get_audioset(
	ROOT_PATH,
	dtype=CACHE_DTYPE,
	truncation=TRUNC if DEBUG else None,
	verbose=VERBOSE
)
descriptive_callback(-1, data)
debug_callback(-1, data)

# transform data
data = transform_data(
	data,
	transforms,
	transform_kwargs,
	[descriptive_callback, debug_callback],
	verbose=VERBOSE
)

# create cache
cache_location = create_cache(
	data,
	class_names=class_names,
	cache_dtype=CACHE_DTYPE,
	cache_root=OUTPUT_PATH,
	cache_name=f'audioset_mono_{int(TARGET_RATE/1000)}khz'
)

# reload cache (opt)
if DEBUG:
	data = reload_cache(cache_location)
	descriptive_callback(len(transforms), data)
	debug_callback(len(transforms), data)
