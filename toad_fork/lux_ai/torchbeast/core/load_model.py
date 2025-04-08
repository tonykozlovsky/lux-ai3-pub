import torch
import time
import os

def loadd_model(model, path, weights_only=True):
    while True:
        try:
            checkpoint_state = torch.load(path, map_location=torch.device("cpu"), weights_only=weights_only)
            adjusted_state_dict = checkpoint_state["model_state_dict"]
            adjusted_state_dict = {
                k.replace("._orig_mod.", "."): v for k, v in adjusted_state_dict.items()
            }
            adjusted_state_dict = {
                k.replace("._orig_mod", ""): v for k, v in adjusted_state_dict.items()
            }
            adjusted_state_dict = {
                k.replace("_orig_mod.", ""): v for k, v in adjusted_state_dict.items()
            }
            adjusted_state_dict = {
                k.replace("_orig_mod", ""): v for k, v in adjusted_state_dict.items()
            }
            if os.getenv("USE_TORCH_COMPILE") == "1":
                adjusted_state_dict = {
                    '_orig_mod.' + k: v for k, v in adjusted_state_dict.items()
                }
            model.load_state_dict(adjusted_state_dict)
            break

        except Exception as e:
            if e == KeyboardInterrupt:
                raise e
            else:
                print(e)
                time.sleep(1)
                continue

