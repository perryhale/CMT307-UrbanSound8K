import jax
import jax.numpy as jnp
import pandas as pd
import scipy as sp


# type: () ->
def dataloader(key, dataset_x, dataset_y, batch_size):
	
	n_samples = dataset_x.shape[0]
	n_batches = int(jnp.ceil(n_samples / batch_size))
	permutation = jax.random.permutation(key, n_samples)
	
	for batch_id in range(n_batches):
		
		start = batch_id * batch_size
		end = min((batch_id + 1) * batch_size, n_samples)
		batch_indices = permutation[start:end]
		
		yield dataset_x[batch_indices], dataset_y[batch_indices]
