import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

eta = 1e-3
n_steps = 128

plt.figure(figsize=(10, 6))
colors = plt.cm.Spectral(np.linspace(0, 1, 16))
markers = 'os^Dv<>p*HXPHd+|_'
i = 0

for factor in np.linspace(0.1, 0.9, 4):
	for rate in np.linspace(0.1, 0.9, 4):
		steps = np.arange(0, n_steps, 10)
		learning_rates = [tf.keras.optimizers.schedules.InverseTimeDecay(
			initial_learning_rate=eta,
			decay_rate=rate,
			decay_steps=factor*n_steps,
			staircase=False 
		)(step) for step in steps]
		plt.plot(steps, learning_rates, label=f"rate={rate:.2f}, factor={factor:.2f}", color=colors[i], marker=markers[i])
		i = i+1

plt.subplots_adjust(right=0.7)
plt.title("InverseTimeDecay Learning Rate Schedule")
plt.xlabel("Steps")
plt.ylabel("Learning Rate")

plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.015), ncol=1)
plt.grid(True)
plt.savefig('lr_schedule.png')
