import scipy

from library.data.io import get_urbansound8k, create_cache, reload_cache
from library.data.pipeline import transform_data, rescale_fn, mono_avg_fn, resample_fn
from library.data.descriptive import wav_stats_fn, plot_distributions

""" https://urbansounddataset.weebly.com/download-urbansound8k.html """
""" Create cache with debug actions and description"""


### arguments

ROOT_PATH = 'data/UrbanSound8K'
OUTPUT_PATH = 'data/'
TARGET_RATE = 24000
CACHE_DTYPE = 'float32'
VERBOSE = True

DEBUG = True
TRUNC = 512*8
EG_IDX = 5


### main

# load data
metadata, class_names, data = get_urbansound8k(
	ROOT_PATH,
	truncation=TRUNC if DEBUG else None,
	verbose=(VERBOSE or DEBUG)
)

# describe data
plot_distributions(data.apply(wav_stats_fn, axis=1), filename=f'{OUTPUT_PATH}/initial_description.png')

###! debug
if DEBUG:
	print(metadata.iloc[EG_IDX,:])
	print(data.iloc[EG_IDX,:]['data'].shape)
	print(data.iloc[EG_IDX,:]['data'].dtype, data)
	scipy.io.wavfile.write(f'{OUTPUT_PATH}/sample{EG_IDX}_initial.wav', data.iloc[EG_IDX,:]['rate'], data.iloc[EG_IDX,:]['data'])

# transform data
data = transform_data(
	data,
	[rescale_fn, mono_avg_fn, resample_fn],
	[{}, {}, {'target_rate' : TARGET_RATE}]
)

# describe data
plot_distributions(data.apply(wav_stats_fn, axis=1), filename=f'{OUTPUT_PATH}/final_description.png')

###! debug
if DEBUG:
	print(metadata.iloc[EG_IDX,:])
	print(data.iloc[EG_IDX,:]['data'].shape)
	print(data.iloc[EG_IDX,:]['data'].dtype, data)
	scipy.io.wavfile.write(f'{OUTPUT_PATH}/sample{EG_IDX}_transform.wav', data.iloc[EG_IDX,:]['rate'], data.iloc[EG_IDX,:]['data'])

# create cache
cache_location = create_cache(
	data,
	class_names=class_names,
	cache_root=OUTPUT_PATH,
	cache_name=f'urbansound8k_mono_{int(TARGET_RATE/1000)}khz'
)

# reload cache (opt)
if DEBUG:
	data = reload_cache(cache_location)
	print(metadata.iloc[EG_IDX,:])
	print(data.iloc[EG_IDX,:]['data'].shape)
	print(data.iloc[EG_IDX,:]['data'].dtype, data)
	scipy.io.wavfile.write(f'{OUTPUT_PATH}/sample{EG_IDX}_reload.wav', data.iloc[EG_IDX,:]['rate'], data.iloc[EG_IDX,:]['data'])
