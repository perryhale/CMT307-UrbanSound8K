import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import RegularGridInterpolator

# get arg
ARG1 = sys.argv[1] if len(sys.argv) > 1 else 'train_transformer_gridsearch.history.pkl'

# Load data
with open(ARG1, 'rb') as f:
    history = pickle.load(f)

# extract values
eta_space = np.array([item['info']['eta'] for item in history[:, 0]])
dpo_space = np.array([item['info']['dropout'] for item in history[0, :]])
val_grid = np.array([[np.max(item['train']['val_accuracy']) for item in row] for row in history])

# interpolate to higher resolution
res = 512
method = 'cubic'
eta_fine, dpo_fine = np.meshgrid(np.linspace(eta_space.min(), eta_space.max(), res), np.linspace(dpo_space.min(), dpo_space.max(), res), indexing='ij')
val_fine = RegularGridInterpolator((eta_space, dpo_space), val_grid, method=method)(np.array([eta_fine.ravel(), dpo_fine.ravel()]).T).reshape(res, res)

# draw high resolution contour
fig, ax = plt.subplots()
contour = ax.contourf(eta_fine, dpo_fine, val_fine, cmap='rainbow', levels=res)
cbar = fig.colorbar(contour)
cbar.set_label('Validation accuracy', rotation=270, labelpad=15)

# draw labels
ax.set_xlabel('Learning rate')
ax.set_ylabel('Dropout rate')

# save and close
plt.savefig(f'{__file__.replace(".py","")}-001.png')
plt.close()

# initialise training plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# iterate over history
high_score = 0
low_score = np.inf
for i in range(history.shape[0]):
    for j in range(history.shape[1]):
		
		# extract values
        model_info = history[i, j]['info']
        train_history = history[i, j]['train']
        
        # plot losses
        ax[0].plot(train_history['loss'], label=f'(eta={model_info["eta"]:.6f}, dpo={model_info["dropout"]:.2f})', linewidth=0.5)
        ax[1].plot(train_history['val_loss'], label=f'(eta={model_info["eta"]:.6f}, dpo={model_info["dropout"]:.2f})', linewidth=0.5)
        
        # determine min and max
        high_score = max(high_score, max(max(train_history['loss']), max(train_history['val_loss'])))
        low_score = min(low_score, min(min(train_history['loss']), min(train_history['val_loss'])))

# draw labels
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Training loss')
ax[0].set_ylim(low_score-0.1, high_score+0.1)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Validation loss')
ax[1].set_ylim(low_score-0.1, high_score+0.1)

# draw legend
#ax[0].legend(fontsize=4, loc='best')
#ax[1].legend(fontsize=4, loc='best')

# save and close
plt.tight_layout()
plt.savefig(f'{__file__.replace(".py","")}-002.png')
