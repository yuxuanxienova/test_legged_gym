
class SimConfig():
  use_gpu_pipeline = True 
  device="cuda"

if __name__ == "__main__":
    sim_cfg = SimConfig
    print(sim_cfg.__dict__)