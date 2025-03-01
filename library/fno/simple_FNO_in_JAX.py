"""

Reference 1D FNO implementation
https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/neural_operators/simple_FNO_in_JAX.ipynb

"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm


# Mathworks (the creators of Matlab) host the original Li et al. dataset in the .mat format
#!wget https://ssd.mathworks.com/supportfiles/nnet/data/burgers1d/burgers_data_R10.mat
data = scipy.io.loadmat("burgers_data_R10.mat")
a, u = data["a"], data["u"]
a.shape
plt.plot(a[0], label="initial condition")
plt.plot(u[0], label="After 1 time unit")
plt.legend()
plt.grid()

# Add channel dimension
a = a[:, jnp.newaxis, :]
u = u[:, jnp.newaxis, :]

# Mesh is from 0 to 2 pi
mesh = jnp.linspace(0, 2 * jnp.pi, u.shape[-1])
plt.plot(mesh, a[0, 0], label="initial condition")
plt.plot(mesh, u[0, 0], label="After 1 time unit")
plt.legend()
plt.grid()
mesh_shape_corrected = jnp.repeat(mesh[jnp.newaxis, jnp.newaxis, :], u.shape[0], axis=0)
a_with_mesh = jnp.concatenate((a, mesh_shape_corrected), axis=1)
a_with_mesh.shape
train_x, test_x = a_with_mesh[:1000], a_with_mesh[1000:1200]
train_y, test_y = u[:1000], u[1000:1200]



in_channels = 2
out_channels = 1
modes = 16
width = 64
activation = jax.nn.relu
n_blocks = 4
loss_fn = mean_squared_error
fno = FNO1d(
	in_channels,
	out_channels,
	modes,
	width,
	activation,
	n_blocks,
	key=jax.random.PRNGKey(0),
)

optimizer = optax.adam(3e-4)
opt_state = optimizer.init(eqx.filter(fno, eqx.is_array))

@eqx.filter_jit
def optimize(model, state, x, y):
	loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x, y)
	val_loss = loss_fn(model, test_x[..., ::32], test_y[..., ::32])
	updates, new_state = optimizer.update(grad, state, model)
	new_model = eqx.apply_updates(model, updates)
	return new_model, new_state, loss, val_loss

loss_history = []
val_loss_history = []

shuffle_key = jax.random.PRNGKey(10)
for epoch in tqdm(range(200)):
	shuffle_key, subkey = jax.random.split(shuffle_key)
	for (batch_x, batch_y) in dataloader(
		subkey,
		train_x[..., ::32],
		train_y[..., ::32],
		batch_size=100,
	):
		fno, opt_state, loss, val_loss = optimize(fno, opt_state, batch_x, batch_y)
		loss_history.append(loss)
		val_loss_history.append(val_loss)

plt.plot(loss_history, label="train loss")
plt.plot(val_loss_history, label="val loss")
plt.legend()
plt.yscale("log")
plt.grid()
plt.plot(test_x[1, 0, ::32], label="Initial condition")
plt.plot(test_y[1, 0, ::32], label="Ground Truth")
plt.plot(fno(test_x[1, :, ::32])[0], label="FNO prediction")
plt.legend()
plt.grid()
plt.plot(fno(test_x[1, :, ::32])[0] - test_y[1, 0, ::32], label="Difference")
plt.legend()

# Zero-Shot superresolution
plt.plot(test_x[1, 0, ::4], label="Initial condition")
plt.plot(test_y[1, 0, ::4], label="Ground Truth")
plt.plot(fno(test_x[1, :, ::4])[0], label="FNO prediction")
plt.legend()
plt.grid()

plt.plot(fno(test_x[1, :, ::4])[0] - test_y[1, 0, ::4], label="Difference")
plt.legend()

# Compute the error as reported in the FNO paper
test_pred = jax.vmap(fno)(test_x)

def relative_l2_norm(pred, ref):
	diff_norm = jnp.linalg.norm(pred - ref)
	ref_norm = jnp.linalg.norm(ref)
	return diff_norm / ref_norm

rel_l2_set = jax.vmap(relative_l2_norm)(test_pred, test_y)
rel_l2_set.shape
jnp.mean(rel_l2_set) # ~1e-2

