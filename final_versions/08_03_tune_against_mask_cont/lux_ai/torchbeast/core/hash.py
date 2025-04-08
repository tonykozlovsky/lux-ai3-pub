
import hashlib
import sys

def out_hash(key, tensor, file, player):
    if key.startswith("info_LOGGING"):
        return
    if key == 'info_GPU1_available_actions_mask_ground_truth_CPU':
        return
    if not isinstance(tensor, torch.Tensor):
        print(key, "UNKNOWN", file=file)
        return
    if key == 'baseline_GPU':
        return
    if key == 'info_GPU1_units_masks_indexes':
        tensor = tensor.view(1, 1, 2, 16, 1)
    if key == 'info_GPU1_units_masks_energy':
        tensor = tensor.view(1, 1, 2, 16, 1)
    if key == 'info_GPU1_units_masks_number':
        tensor = tensor.view(1, 1, 2, 16, 1)
    if key == 'info_GPU1_units_masks_x_cord':
        tensor = tensor.view(1, 1, 2, 16, 1)
    if key == 'info_GPU1_units_masks_y_cord':
        tensor = tensor.view(1, 1, 2, 16, 1)
    if key == 'info_GPU1_available_actions_mask_ground_truth':
        tensor = tensor.view(1, 1, 2, -1, 24, 24)
    if key == 'info_GPU1_units_masks_continues_features':
        tensor = tensor.view(1, 1, 2, 16, -1)
    if key == 'info_GPU1_units_masks_embedding_features':
        tensor = tensor.view(1, 1, 2, 16, -1)
    if key == 'rnn_hidden_state_h_GPU':
        tensor = tensor.unsqueeze(0)
    if key == 'rnn_hidden_state_c_GPU':
        tensor = tensor.unsqueeze(0)
    if key == 'info_rnn_hidden_state_h_GPU':
        tensor = tensor.unsqueeze(0)
    if key == 'info_rnn_hidden_state_c_GPU':
        tensor = tensor.unsqueeze(0)
    if key == 'baseline_GPU':
        tensor = tensor.unsqueeze(0)
    if key == 'prediction_GPU_CPU':
        tensor = tensor.unsqueeze(0)
    if key == 'info_prediction_GPU_CPU':
        tensor = tensor.unsqueeze(0)
    if key == 'info_GPU1_units_masks_additional_features':
        tensor = tensor.unsqueeze(0)

    if key != 'info_GPU1_input_mask':
        try:
            idx = 0 if player == 'player_0' else 1
            tensor = tensor[:, :, idx]
        except Exception:
            print(key, "UNKNOWN", file=file)
            return

    tensor_bytes = tensor.detach().cpu().numpy().tobytes()
    hash_value = hashlib.sha256(tensor_bytes).hexdigest()
    print(key, hash_value, tensor.shape, file=file)

    #if key == 'info_prediction_GPU_CPU':
    #    np.set_printoptions(threshold=np.inf, linewidth=200)
    #    print(key, tensor.cpu().numpy(), file=file)

def out_env_hashes(env_output, file, player, dbg_steps):
    print("steps:", dbg_steps, file=file)
    for key, value in env_output.items():
        if not isinstance(value, dict):
            out_hash(key, value, file, player)
        else:
            for key2, value2 in value.items():
                if not isinstance(value2, dict):
                    out_hash(f"{key}_{key2}", value2, file, player)
                else:
                    for key3, value3 in value2.items():
                        if not isinstance(value3, dict):
                            out_hash(f"{key}_{key2}_{key3}", value3, file, player)
    print('', file=file)
