import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
import random


### setup

ARG1 = sys.argv[1]

with open(ARG1, 'rb') as f:
	history = pickle.load(f)

def print_history_recursive(lvl, h, values=True):
	for key, value in h.items():
		indent = '  ' * lvl
		print(f'{indent}{key}', end='')
		
		if isinstance(value, dict):
			print('{')
			print_history_recursive(lvl+1, value)
			print(f'{indent}}}')
		else:
			print(',')
			if values:
				print(value)

# print statistics
print(history.shape)
print_history_recursive(0, history[1], values=False)

# extract values
folds = [i+1 for i in range(len(history))]
loss = [fold['test']['loss'] for fold in history]
accuracy = [fold['test']['accuracy'] for fold in history]
confusion = [fold['test']['confusion'] for fold in history]
precision = [fold['test']['precision'] for fold in history]
recall = [fold['test']['recall'] for fold in history]
f1 = [fold['test']['f1'] for fold in history]
mean_cfm = np.mean(np.array(confusion), axis=0)

# define colors
cmap = 'Greys'
colors = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']


### write out mean scores

mean_df = pd.DataFrame({k:[v] for k,v in zip(['accuracy', 'precision', 'recall', 'f1'], [np.mean(x) for x in [accuracy, precision, recall, f1]])})
mean_df.to_csv('kfold_mean_scores.csv')


### plot metrics

# init figure
x = np.arange(len(folds))
width = 0.15
fig, ax = plt.subplots(figsize=(7, 5))

# draw bar charts
ax.bar(x - width, accuracy, width, label='Accuracy')
ax.bar(x, precision, width, label='Precision')
ax.bar(x + width, recall, width, label='Recall')
ax.bar(x + 2*width, f1, width, label='F1 Score')

# setup axis
ax.set_xlabel('KFold index')
ax.set_ylabel('Test metric')
ax.set_xticks(x)
ax.set_xticklabels(folds)
ax.legend()
ax.set_ylim(0,1)

# draw annotations
scores = [accuracy]#[accuracy, precision, recall, f1]
min_score = min([min(x) for x in scores])
max_score = max([max(x) for x in scores])
mean_score = np.mean([np.mean(x) for x in scores])

text_kwargs = dict(fontsize=8, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='white', pad=2.0))
line_kwargs = dict(linestyle='dashed', linewidth=0.5)

ax.axhline(max_score, c='g', **line_kwargs)
ax.axhline(mean_score, c='grey', **line_kwargs)
ax.axhline(min_score, c='r', **line_kwargs)

ax.text(0, max_score+0.15, f'Max accuracy: {max_score:.4f}', c='g', **text_kwargs)
ax.text(0, max_score+0.10, f'Mean accuracy: {mean_score:.4f}', c='black', **text_kwargs)
ax.text(0, max_score+0.05, f'Min accuracy: {min_score:.4f}', c='r', **text_kwargs)

# save and close
plt.tight_layout()
plt.savefig(f'{__file__.replace(".py","")}-001.png')
plt.close()


### plot mean cfm

def plot_confusion_matrix(ax, cfm, labels, title='', precision=2, cmap='viridis', colorbar=False):
	
	# draw 
	im = ax.matshow(cfm, cmap=cmap)
	if colorbar: plt.colorbar(im, ax=ax, shrink=0.8)
	
	# setup axis
	ax.set_xlabel(f'{title}Predicted class', fontsize=12)
	ax.set_ylabel(f'{title}Actual class', fontsize=12)
	ax.set_xticks(np.arange(len(labels)))
	ax.set_yticks(np.arange(len(labels)))
	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)
	
	# annotate values
	for i in range(len(labels)):
		for j in range(len(labels)):
			
			# adjust font color by cell luminance
			# https://stackoverflow.com/a/596243
			cell_value = cfm[i, j]
			cell_color = im.cmap(cell_value / np.max(cfm))
			luminance = 0.2126 * cell_color[0] + 0.7152 * cell_color[1] + 0.0722 * cell_color[2]
			text_color = 'white' if luminance < 0.75 else 'black'
			
			ax.text(j, i, f'{cfm[i,j]:.1f}', ha='center', va='center', color=text_color, fontsize=8)
	
	return ax

# for c in ['Greys']:#colors:
	# fig, ax = plt.subplots(figsize=(7, 7))
	# ax = plot_confusion_matrix(ax, mean_cfm, range(len(mean_cfm)), title='Mean Test confusion', cmap=c)
	# plt.savefig(f'{cmap}.png')
	# plt.close()

fig, ax = plt.subplots(figsize=(7, 7))
ax = plot_confusion_matrix(ax, mean_cfm, range(1,len(mean_cfm)+1), title='(Mean) ', cmap=cmap)
plt.savefig(f'{__file__.replace(".py","")}-002.png')
plt.close()


### plot all cfm

# row-normalise confusion matrices
conf_norm = [cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis] for cfm in confusion]
max_cfm = np.max([np.max(x) for x in conf_norm])
min_cfm = np.min([np.min(x) for x in conf_norm])

# init figure
fig, axis = plt.subplots(nrows=1, ncols=len(confusion), figsize=(22, 3))

# plot cfms
for i, (ax, cfm) in enumerate(zip(axis, conf_norm)):
	im = ax.matshow(cfm, cmap=cmap, vmin=min_cfm, vmax=max_cfm)
	#im = ax.matshow(cfm, cmap=cmap)
	ax.set_title(f'Fold {i+1}', fontsize=10)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xlabel('Predicted class', fontsize=8)
	if i==0:
		ax.set_ylabel('Actual class', fontsize=8)

# save and close
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=None, hspace=None)
plt.savefig(f'{__file__.replace(".py","")}-003.png')
plt.close()
