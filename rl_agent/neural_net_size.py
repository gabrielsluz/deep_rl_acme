import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

# from acme.jax import networks as networks_lib
# from acme.jax import utils


# def network_fn(obs):
#     num_actions = 8
#     network = hk.Sequential([
#         hk.Conv2D(output_channels=16, kernel_shape=[4, 4], stride=2, padding='valid'),
#         jax.nn.relu,
#         hk.Conv2D(output_channels=16, kernel_shape=[3, 3], padding='valid'),
#         jax.nn.relu,
#         hk.Flatten(),
#         hk.Linear(64),
#         jax.nn.relu,
#         hk.Linear(num_actions)
#     ])
#     x = obs
#     x = network(x)
#     return x
def network_fn(obs):
    network = hk.Sequential([
        hk.Conv2D(output_channels=6, kernel_shape=[4, 4], stride=1, padding='VALID'),
        jax.nn.relu,
        hk.MaxPool(window_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'),
        hk.Conv2D(output_channels=16, kernel_shape=[4, 4], stride=1, padding='VALID'),
        jax.nn.relu,
        hk.MaxPool(window_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'),
        hk.Flatten(),
        hk.Linear(120),
        jax.nn.relu,
        hk.Linear(84),
        jax.nn.relu,
        hk.Linear(8)
    ])
    x = obs
    x = network(x)
    return x

dummy_obs = np.zeros((1,16,16,4))
mlp = hk.without_apply_rng(hk.transform(network_fn))
params = mlp.init(0, dummy_obs)

total_params = 0

for key in params.keys():
    for inner_key in params[key].keys():
        print(key, inner_key, params[key][inner_key].shape)
        total_params += np.prod(params[key][inner_key].shape)
print('Total number of params = ', total_params)
