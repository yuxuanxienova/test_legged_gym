
class SimConfig():
  dt =  0.005#1/240.  # [s]
  substeps = 1
  gravity = [0., 0. ,-9.81]  # [m/s^2]
  use_gpu_pipeline = True
  device="cuda"
  class physx:
      num_threads = 10
      solver_type = 1  # 0: pgs, 1: tgs
      num_position_iterations = 4
      num_velocity_iterations = 0
      contact_offset = 0.01  # [m]
      rest_offset = 0.0   # [m]
      bounce_threshold_velocity = 0.5 #0.5 [m/s]
      max_depenetration_velocity = 1.0
      max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
      default_buffer_size_multiplier = 5
      contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

if __name__ == "__main__":
    sim_cfg = SimConfig
    print(sim_cfg.__dict__)