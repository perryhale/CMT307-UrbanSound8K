import jax
import jax.numpy as jnp


def mean_squared_error(model, x, y):
	y_pred = jax.vmap(model)(x)
	loss = jnp.mean(jnp.square(y_pred - y))
	return loss
