import jax
import jax.numpy as jnp

# no need params if using openmpi
jax.distributed.initialize()

print("total devices: %s, devices per task: %s" % (jax.device_count(), jax.local_device_count()))

xs = jnp.ones(jax.local_device_count())

# Computes a reduction (sum) across all devices of x
# and broadcast the result, in y, to all devices.
# If x=[1] on all devices and we have 16 devices,
# the result is y=[16] on all devices.

y = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(xs)
print(y)
