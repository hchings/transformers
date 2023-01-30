# script.py
import jax
from jax.experimental.pjit import pjit
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as P
import numpy as np
import argparse
import timeit


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, choices=['pmap', 'pjit'], default='pmap')
args = parser.parse_args()


# Init data
x = np.random.randn(32, 1024).astype(np.float32)
W = np.random.randn(1024, 8).astype(np.float32)


def step(x, W):
    return jax.lax.dot(x, W)


# Compute pmap or pjit functions
# Preload batch data and model parameters onto the devices as ShardedDeviceArrays
if args.mode == 'pmap':
    p_step = jax.pmap(step, axis_name='batch')
    print("pmap mode. backend=%s, device_count=%s, local_device_count=%s" % (jax.lib.xla_bridge.get_backend(),
                                                                          jax.device_count(),
                                                                          jax.local_device_count()))
    x = np.reshape(x, (jax.local_device_count(), -1, x.shape[1]))
    print("x shape:", x.shape)

    # Gets correct device order that matches pmap
    devices = jax.lib.xla_bridge.get_backend().get_default_device_assignment(jax.device_count())
    x = jax.device_put_sharded(list(x), devices)
    W = jax.device_put_replicated(W, devices)
else:
    # ===== DP only =====
    mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape(jax.local_device_count(), ), ['dp'])
    jax.experimental.maps.thread_resources.env = (
        jax.experimental.maps.ResourceEnv(physical_mesh=mesh, loops=())
    )
    p_step = pjit(step, in_axis_resources=(P('dp'), None), out_axis_resources=P('dp'))

    # Map batch and weights to devices
    p_init = pjit(lambda x, W: (x, W), in_axis_resources=(P('dp'), None), out_axis_resources=(P('dp'), None))
    x, W = p_init(x, W)

    # ===== DP & MP =====
    # mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape(jax.local_device_count(), 1), ['dp', 'mp'])
    # jax.experimental.maps.thread_resources.env = (
    #     jax.experimental.maps.ResourceEnv(physical_mesh=mesh, loops=())
    # )
    # p_step = pjit(step, in_axis_resources=(P('dp'), P('mp', None)), out_axis_resources=P('dp'))
    #
    # # Map batch and weights to devices
    # p_init = pjit(lambda x, W: (x, W), in_axis_resources=(P('dp'), P('mp', None)), out_axis_resources=(P('dp'), P('mp', None)))
    # x, W = p_init(x, W)

# Warmup for initial compilation
p_step(x, W).block_until_ready()

# Time
iterations = 1000
avg = timeit.timeit(lambda: p_step(x, W).block_until_ready(), number=iterations) / iterations
print('Estimated Time:', avg, 'per itr')