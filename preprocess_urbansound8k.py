import numpy as np
import pandas as pd
import scipy as sp
import base64
from library.data import get_urbansound8k, reload_cache

import matplotlib.pyplot as plt
from collections import Counter

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


### main

# load and preprocess
metadata, class_names, data = get_urbansound8k(
	ROOT_PATH,
	target_rate=TARGET_RATE,
	###! debug
	truncation=TRUNC if DEBUG else None,
	verbose=(VERBOSE or DEBUG)
)

###! debug
if DEBUG:
	example_index = 5#int(np.random.uniform(0,1) * TRUNC)
	print(metadata.iloc[example_index,:])
	print(data['data'][example_index].shape)
	print(data['data'][example_index].dtype, data['data'][example_index])
	sp.io.wavfile.write(f'{OUT_PATH}/data0_{int(TARGET_RATE/1000)}khz_float64.wav', data['rate'][example_index], data['data'][example_index])
	with open(f'{OUT_PATH}/raw_size_ref.pkl', 'wb') as f:
		import pickle
		pickle.dump(data, f)

# cache class names
class_names_df = pd.DataFrame.from_dict(class_names, orient='index')
class_names_df.to_csv(f'{OUT_PATH}/urbansound8k_classes.csv', header=['name'], index_label='id')

# cache data
data['data'] = data['data'].apply(lambda x : base64.b64encode(x.astype(CACHE_DTYPE).tobytes()).decode('utf-8'))
data.to_csv(f'{OUT_PATH}/urbansound8k_mono_{int(TARGET_RATE/1000)}khz_{CACHE_DTYPE}.csv', index=False) # [rate, data, fold, class]

###! debug
if DEBUG:
	data = reload_cache(f'{OUT_PATH}/urbansound8k_mono_{int(TARGET_RATE/1000)}khz_{CACHE_DTYPE}.csv')
	print(data['data'][example_index].dtype, data['data'][example_index])
	for i, row in data.iterrows():
		sp.io.wavfile.write(f'{OUT_PATH}/data{i}_{int(TARGET_RATE/1000)}khz_{CACHE_DTYPE}.wav', row['rate'], row['data'])
	print(data['data'][0].dtype)
	print(data['data'].apply(lambda x : np.min(x).item()).min())
	print(data['data'].apply(lambda x : np.max(x).item()).max())
	print(data)

###! debug notes
# 48KHz mono wav float64 -> floatXX -> bytes -> base64 string
# b64(f64) > f64 > b64(f32)
# 14.5MB > 10.9MB (pkl) > 7.2MB @8samples

# f = lambda x, l : np.pad(x, (0,l-x.shape[-1])) if x.shape[-1]<l else x[:l]



### plot sequence length distribution

sequence_lengths = [len(x) for x in data['data']]
sizes, freqs = zip(*sorted(Counter(sequence_lengths).items(), key=lambda t: -t[0]))
probs = [f / float(len(data['data'])) for f in freqs]
print(sizes)
print(freqs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

#ax1.scatter(sizes, freqs, color='r', marker='x')
for i in range(len(sizes)): ax1.vlines(sizes[i], 0, freqs[i], colors='red')
ax1.set_xlabel('Sequence length')
ax1.set_ylabel('Count')
ax1.grid()

#ax2.scatter(sizes, probs, marker='x')
for i in range(len(sizes)): ax2.vlines(sizes[i], 0, probs[i], colors='C0')
ax2.set_xlabel('Sequence length')
ax2.set_ylabel('Probability')
ax2.grid()

#plt.show()
plt.savefig(f'{OUT_PATH}/preprocess_urbansound8k_sequences.png')
