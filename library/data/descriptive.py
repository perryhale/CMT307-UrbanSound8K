import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def wav_stats_fn(row):
	""" Extract dictionary of audio statistics
	# type: (pd.Series) -> pd.Series
	"""
	
	# unpack
	rate = row['rate']
	data = row['data']
	
	# extract statistics
	stats = {}
	stats['channels'] = data.shape[1] if len(data.shape) > 1 else 1
	stats['sample_width'] = data.itemsize
	stats['sample_rate'] = rate
	stats['n_samples'] = data.shape[0]
	stats['duration'] = stats['n_samples'] / stats['sample_rate']
	stats['bit_rate'] = stats['sample_rate'] * stats['channels'] * stats['sample_width'] * 8
	
	return stats


def plot_distributions(stats_column, filename='data_description.png'):
	""" Plot distributions of audio statistics
	# type: (pd.Series|List[Dict[str:float|int]], str) -> None
	"""
	
	# extract values
	durations = [stats['duration'] for stats in stats_column]
	channels = [stats['channels'] for stats in stats_column]
	sample_widths = [stats['sample_width'] for stats in stats_column]
	sample_rates = [stats['sample_rate'] for stats in stats_column]
	bit_rates = [stats['bit_rate'] for stats in stats_column]
	lengths = [stats['n_samples'] for stats in stats_column]
	
	# init figure
	fig, axes = plt.subplots(3, 2, figsize=(14, 10))
	
	# plotting function
	def plot_hist_or_value(ax, data, title, xlabel, ylabel, color):
		unique_data = np.unique(data)
		ax.set_title(title)
		ax.set_xlabel(xlabel)
		if unique_data.shape[0] == 1:
			ax.text(0.5, 0.5, f'{unique_data[0]}', ha='center', va='center', fontsize=15, color='black')
			ax.set_xticks([], [])
			ax.set_yticks([], [])
		else:
			ax.hist(data, bins=min(20, np.unique(data).shape[0]), color=color, edgecolor='black', alpha=0.7)
			ax.set_ylabel(ylabel)
			ax.grid()
	
	# plot distributions
	plot_hist_or_value(axes[0, 0], durations, 'Duration distribution', 'Duration (seconds)', 'Frequency', 'skyblue')
	plot_hist_or_value(axes[0, 1], channels, 'Channels distribution', 'Channels', 'Frequency', 'lightgreen')
	plot_hist_or_value(axes[1, 0], sample_widths, 'Sample-width distribution', 'Sample width (bytes)', 'Frequency', 'orange')
	plot_hist_or_value(axes[1, 1], sample_rates, 'Sample-rate distribution', 'Sample rate (samples/second)', 'Frequency', 'orange')
	plot_hist_or_value(axes[2, 0], bit_rates, 'Bit-rate distribution', 'Bit rate (bytes/second)', 'Frequency', 'green')
	plot_hist_or_value(axes[2, 1], lengths, 'Num-samples distribution', 'Samples', 'Frequency', 'red')
	
	# finalize
	plt.tight_layout()
	plt.savefig(filename)
	plt.close()


def plot_tokenized_sample(data_x, prefix='unnamed', key=None):
	""" Plot randomly selected 2d sample and header
	# type: (np.ndarray, str, int) -> None
	"""
	
	# select sample
	x_sample = data_x[np.random.default_rng(seed=key).integers(0, len(data_x))]
	
	# plot full sample
	plt.figure(figsize=(4,10))
	plt.imshow(x_sample)
	plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
	plt.savefig(f'{prefix}-001.png')
	plt.close()
	
	# plot sample header
	plt.figure(figsize=(10,3))
	plt.imshow(x_sample[:16])
	plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
	plt.savefig(f'{prefix}-002.png')
	plt.subplots_adjust()
	plt.close()
