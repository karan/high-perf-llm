import jax
import s02_timing_util

for MATRIX_DIM in (64, 128, 256, 512, 1024, 2048, 4096):
  NUM_MATRICES = 2**28 // (MATRIX_DIM ** 2)
  STEPS = 10

  A = jax.numpy.ones( (NUM_MATRICES, MATRIX_DIM, MATRIX_DIM), dtype=jax.numpy.bfloat16 )
  B = jax.numpy.ones( (NUM_MATRICES, MATRIX_DIM, MATRIX_DIM), dtype=jax.numpy.bfloat16 )

  num_bytes = A.size * 2 # bfp16
  total_num_bytes_crossing_to_hbm = 3 * num_bytes

  total_num_flops = (2 * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM) * NUM_MATRICES

  @jax.jit
  def f(A, B):
    return jax.lax.batch_matmul(A, B)

  average_time_sec_jitted = s02_timing_util.simple_timeit(f, A, B, task="jit_f") / 1000

  print(f"matrix dim: {MATRIX_DIM}")
  print(f"arithmetic intensity: {total_num_flops / total_num_bytes_crossing_to_hbm}")
  print(f"jit: {average_time_sec_jitted}, terraflops per second {total_num_flops/average_time_sec_jitted / 10**12}, gigabytes per second {total_num_bytes_crossing_to_hbm/average_time_sec_jitted/10**9}")
  print("\n\n\n")
