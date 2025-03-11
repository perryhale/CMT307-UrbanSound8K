import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def wav_stats_fn(row):
	rate, data = row['rate'], row['data']
	stats = {}
	stats['channels'] = data.shape[1] if len(data.shape) > 1 else 1
	stats['sample_width'] = data.itemsize
	stats['sample_rate'] = rate
	stats['num_samples'] = data.shape[0]
	stats['duration_seconds'] = stats['num_samples'] / stats['sample_rate']
	stats['bit_rate'] = stats['sample_rate'] * stats['channels'] * stats['sample_width'] * 8
	return stats


def plot_distributions(stats_list, filename='data_description.png'):
	
	# extract values
	durations = [stats['duration_seconds'] for stats in stats_list]
	channels = [stats['channels'] for stats in stats_list]
	sample_widths = [stats['sample_width'] for stats in stats_list]
	sample_rates = [stats['sample_rate'] for stats in stats_list]
	bit_rates = [stats['bit_rate'] for stats in stats_list]
	lengths = [stats['num_samples'] for stats in stats_list]
	
	# init figure
	fig, axes = plt.subplots(3, 2, figsize=(14, 10))
	
	# plotting function
	def plot_or_display_value(ax, data, title, xlabel, ylabel, color):
		ax.set_title(title)
		ax.set_xlabel(xlabel)
		if np.unique(data).shape[0] == 1:
			unique_value = np.unique(data)[0]
			ax.text(0.5, 0.5, f'{unique_value}', ha='center', va='center', fontsize=15, color='black')
			ax.set_xticks([], [])
			ax.set_yticks([], [])
		else:
			ax.hist(data, bins=min(20, np.unique(data).shape[0]), color=color, edgecolor='black', alpha=0.7)
			ax.set_ylabel(ylabel)
			ax.grid()
	
	# plot durations
	plot_or_display_value(axes[0, 0], durations, 'Duration Distribution', 'Duration (seconds)', 'Count', 'skyblue')
	
	# plot channels
	plot_or_display_value(axes[0, 1], channels, 'Channels Distribution', 'Channels', 'Count', 'lightgreen')
	
	# plot sample width
	plot_or_display_value(axes[1, 0], sample_widths, 'Sample Width Distribution', 'Sample width (bytes)', 'Count', 'orange')
	
	# plot sample rate
	plot_or_display_value(axes[1, 1], sample_rates, 'Sample Rate Distribution', 'Sample rate (samples/second)', 'Count', 'orange')
	
	# plot bit rate
	plot_or_display_value(axes[2, 0], bit_rates, 'Bit Rate Distribution', 'Bit rate (bps)', 'Count', 'green')
	
	# plot n samples
	plot_or_display_value(axes[2, 0], lengths, 'Num-samples Distribution', 'Bit rate (bps)', 'Count', 'green')
	
	# finalize
	axes[2, 1].axis('off')
	plt.tight_layout()
	plt.savefig(filename)
