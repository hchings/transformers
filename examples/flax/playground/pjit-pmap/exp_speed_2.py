# script2.py
from typing import Tuple, Any
import timeit
import argparse
import jax
from jax.experimental.pjit import pjit
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as P
import numpy as np
import flax.linen as nn


class Model(nn.Module):
    enc_args: Any
    tfm_args: Any
    dec_args: Any

    @nn.compact
    def __call__(self, x):
        x = jax.vmap(Encoder(**self.enc_args), 1, 1)(x)

        old_shape = x.shape[1:-1]
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = Transformer(**self.tfm_args)(x)
        x = x.reshape(x.shape[0], *old_shape, x.shape[-1])

        x = jax.vmap(Decoder(**self.dec_args), 1, 1)(x)
        return x


def block(x, depth):
    skip = x
    if skip.shape[-1] != depth:
        skip = nn.Conv(depth, [1, 1], use_bias=False)(skip)
    x = nn.Sequential([
        nn.GroupNorm(),
        nn.elu,
        nn.Conv(depth, [3, 3]),
        nn.GroupNorm(),
        nn.elu,
        nn.Conv(depth, [3, 3])
    ])(x)
    return skip + 0.1 * x


class Encoder(nn.Module):
    depths: Tuple
    blocks: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.depths[0], [3, 3])(x)
        for i in range(1, len(self.depths)):
            x = nn.avg_pool(x, (2, 2), strides=(2, 2))
            for _ in range(self.blocks):
                x = block(x, self.depths[i])
        return x


class Decoder(nn.Module):
    depths: Tuple
    blocks: int

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.depths) - 1):
            for _ in range(self.blocks):
                x = block(x, self.depths[i])
            x = jax.image.resize(x, (x.shape[0], 2 * x.shape[1], 2 * x.shape[2], x.shape[3]),
                                 method='nearest')
        x = nn.Conv(3, [3, 3])(x)
        return x


class Transformer(nn.Module):
    hidden_dim: int
    num_heads: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        for _ in range(self.num_layers):
            x = TransformerBlock(self.hidden_dim, self.num_heads)(x)
        return x


class TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(num_heads=self.num_heads)(x)
        x = x + h

        h = nn.LayerNorm()(x)
        h = nn.Sequential([
            nn.Dense(4 * self.hidden_dim),
            nn.gelu,
            nn.Dense(self.hidden_dim)
        ])(h)
        x = x + h
        return x

def print_model_size(params, name=''):
    model_params_size = jax.tree_util.tree_map(lambda x: x.size, params)
    total_params_size = sum(jax.tree_util.tree_flatten(model_params_size)[0])
    print('model parameter count:', total_params_size)


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, choices=['pmap', 'pjit'], default='pmap')
parser.add_argument('-d', '--device_count', type=int, default=0)
args = parser.parse_args()


# Init data
x = np.random.randn(32, 100, 16, 16, 3).astype(np.float32)
model = Model(enc_args=dict(depths=[64, 128, 256], blocks=2),
              tfm_args=dict(hidden_dim=512, num_heads=8, num_layers=8),
              dec_args=dict(depths=[256, 128, 64], blocks=2))
variables = model.init(rngs=jax.random.PRNGKey(0), x=x)
print_model_size(variables)

def step(x, variables):
    # shaped array
    return model.apply(variables, x)

# Compute pmap or pjit functions
# Preload batch data and model parameters onto the devices as ShardedDeviceArrays
if args.mode == 'pmap':
    p_step = jax.pmap(step, axis_name='batch')

    device_count =jax.local_device_count()
    if args.device_count:
        device_count = args.device_count

    print("using %s devices" % device_count)

    # x = np.reshape(x, (jax.local_device_count(), -1, *x.shape[1:]))
    x = np.reshape(x, (device_count, -1, *x.shape[1:]))

    # Gets correct device order that matches pmap
    # devices = jax.lib.xla_bridge.get_backend().get_default_device_assignment(jax.device_count())
    devices = jax.lib.xla_bridge.get_backend().get_default_device_assignment(device_count)
    x = jax.device_put_sharded(list(x), devices)
    variables = jax.device_put_replicated(variables, devices)
else:
    mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape(jax.local_device_count(),), ['dp'])
    jax.experimental.maps.thread_resources.env = (
        jax.experimental.maps.ResourceEnv(physical_mesh=mesh, loops=())
    )
    p_step = pjit(step, in_axis_resources=(P('dp'), None), out_axis_resources=P('dp'))

    # Map batch and weights to devices
    p_init = pjit(lambda x, variables: (x, variables), in_axis_resources=(P('dp'), None), out_axis_resources=(P('dp'), None))
    x, variables = p_init(x, variables)

# Warmup for initial compilation
p_step(x, variables).block_until_ready()

# Time
iterations = 100
avg = timeit.timeit(lambda: p_step(x, variables).block_until_ready(), number=iterations) / iterations
print('Estimated Time:', avg, 'per itr')