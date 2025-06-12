from functools import partial
import numpy as np
import jax
import s02_timing_util

MATRIX_SIZE = 16384
BATCH_PER_CHIP = int(MATRIX_SIZE / jax.device_count())
LAYERS = 4

ACTIVATION = jax.numpy.ones( (BATCH_PER_CHIP*jax.device_count(), MATRIX_SIZE), dtype=jax.numpy.bfloat16 )
WEIGHTS = [jax.numpy.ones( (MATRIX_SIZE, MATRIX_SIZE), dtype=jax.numpy.bfloat16 ) for i in range(LAYERS)]

mesh = jax.sharding.Mesh(jax.devices(), ('myaxis'))
activation_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('myaxis', None))
weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, 'myaxis'))

ACTIVATION = jax.device_put(ACTIVATION, activation_sharding)
WEIGHTS = [jax.device_put(w, weight_sharding) for w in WEIGHTS]

@jax.jit
def matmul(_act, _weights):
  for _w in _weights:
    _act = _act @ _w
  return _act

average_time_ms = s02_timing_util.simple_timeit(matmul, ACTIVATION, WEIGHTS, task="matmul")
