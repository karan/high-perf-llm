import numpy as np
import jax

A = jax.numpy.ones((1024, 1024))

mesh = jax.sharding.Mesh(np.reshape(jax.devices(), (2, 4)), ["myaxis1", "myaxis2"])
p = jax.sharding.PartitionSpec("myaxis1", "myaxis2")
sharding = jax.sharding.NamedSharding(mesh, p)
sharded_A = jax.device_put(A, sharding)

print(f"{sharded_A.shape=} {sharded_A.addressable_shards[0].data.shape=}")

jax.debug.visualize_array_sharding(sharded_A)
