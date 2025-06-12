from functools import partial
import numpy as np
import jax
import s02_timing_util

MATRIX_SIZE = 16384

A = jax.numpy.ones( (MATRIX_SIZE, MATRIX_SIZE), dtype=jax.numpy.bfloat16 )

mesh = jax.sharding.Mesh(jax.devices(), ('myaxis'))
sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('myaxis'))
unsharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))

A = jax.device_put(A, sharded)

@partial(jax.jit, out_shardings = unsharded)
def unshard_array(input):
  return input

average_time_ms = s02_timing_util.simple_timeit(unshard_array, A, task="unshard_array")
achieved_bandwidth_GB_s = (A.size * 2 / 1e9) / (average_time_ms / 1e3)

print(f"achieved bandwidth: {achieved_bandwidth_GB_s} GB/s")

sharded_A = jax.device_put(A, sharded)
unsharded_A = unshard_array(A)

jax.debug.visualize_array_sharding(sharded_A)
jax.debug.visualize_array_sharding(unsharded_A)
