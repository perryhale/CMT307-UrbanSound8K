from collections import Counter
import base64
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from library.data.io import get_urbansound8k, create_cache, reload_cache
from library.data.pipeline import rescale_fn, mono_avg_fn, resample_fn

""" https://urbansounddataset.weebly.com/download-urbansound8k.html """
""" Create cache with debug actions """


### arguments

ROOT_PATH = 'data/UrbanSound8K'
OUT_PATH = 'data/'
TARGET_RATE = 24000
CACHE_DTYPE = 'float32'
VERBOSE = True

DEBUG = True
TRUNC = 512
EG_IDX = 5


### functions

def write_wav_dbg_fn(rate, data, metadata=None, path='sample.wav'):
	if metadata is not None:
		print(metadata.iloc[EG_IDX,:])
	print(data.shape)
	print(data.dtype, data)
	sp.io.wavfile.write(path, rate, data)


### main

# load data
metadata, class_names, data = get_urbansound8k(
	ROOT_PATH,
	truncation=TRUNC if DEBUG else None,
	verbose=(VERBOSE or DEBUG)
)

# define data transformations
transforms = [
	rescale_fn,
	mono_avg_fn,
	resample_fn
]
transform_kwargs = [
	{},
	{},
	{'target_rate' : TARGET_RATE}
]

# apply data transformations
for transform, kwargs in zip(transforms, transform_kwargs):
	data = data.apply(transform, **kwargs, axis=1)
	if (VERBOSE or DEBUG):
		print(data)
	if DEBUG:
		write_wav_dbg_fn(
			data['rate'][EG_IDX],
			data['data'][EG_IDX],
			metadata=metadata,
			path=f'{OUT_PATH}/sample{EG_IDX}_transformed.wav'
		)

# create cache
cache_location = create_cache(
	data,
	class_names=class_names,
	cache_root=OUT_PATH,
	cache_name=f'urbansound8k_mono_{int(TARGET_RATE/1000)}khz_{CACHE_DTYPE}'
)

# test reload cache (opt)
if DEBUG:
	data = reload_cache(cache_location)
	write_wav_dbg_fn(
		data['rate'][EG_IDX],
		data['data'][EG_IDX],
		metadata=metadata,
		path=f'{OUT_PATH}/sample{EG_IDX}_reloaded.wav'
	)

###! DEBUG notes
# 48KHz mono wav float64 -> floatXX -> bytes -> base64 string
# b64(f64) > f64 > b64(f32)
# 14.5MB > 10.9MB (pkl) > 7.2MB @8samples


### plot sequence length distribution

sequence_lengths = [len(x) for x in data['data']]
sizes, freqs = zip(*sorted(Counter(sequence_lengths).items(), key=lambda t: -t[0]))
probs = [f / float(len(data['data'])) for f in freqs]
print(sizes)
print(freqs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
for i in range(len(sizes)): ax1.vlines(sizes[i], 0, freqs[i], colors='red') ###! values are too sparse for bar rendering
ax1.set_xlabel('Sequence length')
ax1.set_ylabel('Count')
ax1.grid()
for i in range(len(sizes)): ax2.vlines(sizes[i], 0, probs[i], colors='C0')
ax2.set_xlabel('Sequence length')
ax2.set_ylabel('Probability')
ax2.grid()
#plt.show()
plt.savefig(f'{OUT_PATH}/preprocess_urbansound8k_sequences.png')
