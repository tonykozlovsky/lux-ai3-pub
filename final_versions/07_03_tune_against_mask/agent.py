try:
    DEBUG__ = False
    import os
    os.environ["OMP_NUM_THREADS"] = "1" if DEBUG__ else "2" # for use 1.6 available cores ???????/
    import copy
    import hashlib
    import numpy as np
    import sys
    import traceback
    from pathlib import Path

    import torch
    import secrets
    import time
    import einops

    from library import rotation
    from lux_ai.lux_gym import create_env
    from lux_ai.nns import create_model
    from lux_ai.nns.models import DictSeparateActorResn
    from lux_ai.torchbeast.core import prof
    from lux_ai.utility_constants import MODEL_PATH
    from lux_ai.utils import flags_to_namespace
    from omegaconf import DictConfig, OmegaConf


except Exception as e:
    print(e, file=sys.stderr)
    print("Error import", file=sys.stderr)
    pass

MIN_OVERAGE_TIME_FOR_AUG = 30 # allow 30s overtime before switching to one player inference

use_weak_model = False # false for internal testing, True for kaggle
strong_all_rounds = True
weak_enables_mask = False

weak_model_probability = 0.8
model_name = '300000000_weights.pt'
model_power_koef = 0.46 # 1. is full power and 0. is random actions

# 0.5 high

class Agent():

    def out_hash(self, key, tensor, file):
        if key.startswith("info_LOGGING"):
            return
        if key == 'info_GPU1_available_actions_mask_ground_truth_CPU':
            return
        if key == 'baseline_GPU':
            return
        if not isinstance(tensor, torch.Tensor):
            print(key, "UNKNOWN", file=file)
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
        if key == 'info_GPU1_units_masks_one_hot_encoded_embedding_features':
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
                idx = 0 if self.player == 'player_0' else 1
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

    def out_env_hashes(self, env_output, file, dbg_steps):
        print("steps:", dbg_steps, file=file)
        for key, value in env_output.items():
            if not isinstance(value, dict):
                self.out_hash(key, value, file)
            else:
                for key2, value2 in value.items():
                    if not isinstance(value2, dict):
                        self.out_hash(f"{key}_{key2}", value2, file)
                    else:
                        for key3, value3 in value2.items():
                            if not isinstance(value3, dict):
                                self.out_hash(f"{key}_{key2}_{key3}", value3, file)
        print('', file=file)

    def prepare(self):
        try:
            dirname = os.path.dirname(__file__)


            #print(dirname, file=sys.stderr)

            folder_name = MODEL_PATH

            if DEBUG__:
                self.file = open(f"outputs/1debug_hashes_agent_{self.player}.txt", "w")

            actor_index = 0
            config_path = os.path.join(dirname, folder_name, "config.yaml")

            flags = OmegaConf.load(config_path)

            if use_weak_model and weak_enables_mask:
                flags.enable_sap_masks = True

            #overwrite
            self.data_augmentation = True

            if DEBUG__:
                flags.flip_axes = False
                self.data_augmentation = False

            if 'flip_learning' not in flags:
                self.flip_learning = True
            else:
                self.flip_learning = flags.flip_learning

            flags = flags_to_namespace(OmegaConf.to_container(flags))

            flags.disable_wandb = True
            flags.actor_device = torch.device("cuda:0") if (torch.cuda.is_available() and not DEBUG__) else torch.device("cpu")

            flags.learner_device = flags.actor_device
            flags.num_actors = 1
            flags.n_actor_envs = 2 if (self.data_augmentation and not self.flip_learning) else 1
            flags.batch_size = flags.n_actor_envs

            flags.kaggle = True

            if use_weak_model:
                value = secrets.randbelow(10**9) / 10**9  # Generate a random float in [0,1)
                self.current_is_weak = value < weak_model_probability
                self.is_weak = self.current_is_weak
            else:
                value = 228
                self.current_is_weak = False
                self.is_weak = False
            if not self.current_is_weak:
                print("HAS_MODEL", value, file=sys.stderr)

            model_file = os.path.join(dirname, folder_name, model_name)

            checkpoint_state = torch.load(model_file, map_location=flags.actor_device, weights_only=True)

            self.actor_model = create_model(flags, flags.actor_device, teacher_model_flags=None, is_teacher_model=False)

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

            # Load the state_dict into the uncompiled model
            self.actor_model.load_state_dict(adjusted_state_dict)

            #self.actor_model.load_state_dict(checkpoint_state["model_state_dict"], strict=False)
            self.actor_model.eval()


            self.env = create_env(actor_index, flags, device=flags.actor_device, teacher_flags=None)


            lstm_size = (128, 24, 24) if flags.enable_lstm else (1, 1, 1)

            self.prev_hidden_state_h = torch.zeros(flags.n_actor_envs, 2, *lstm_size, device=flags.actor_device, dtype=torch.float32)
            self.prev_hidden_state_c = torch.zeros(flags.n_actor_envs, 2, *lstm_size, device=flags.actor_device, dtype=torch.float32)
            self.prev_prediction = torch.zeros(flags.n_actor_envs, 2, 3, 24, 24, device=flags.actor_device, dtype=torch.float32)


        except Exception as e:
            print(e, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("Error loading model, using random agent", file=sys.stderr)
            pass



    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.prev_actions_taken = None
        self.dbg_steps = 0

        self.prepare()


    def swap(self, arr):
        arr[[0, 1]] = arr[[1, 0]]

    def flip_obs(self, obs):
        #print('obs', obs.keys(), file=sys.stderr)
        #assert str(obs.keys()) == "dict_keys(['units', 'units_mask', 'sensor_mask', 'map_features', 'relic_nodes', 'relic_nodes_mask', 'team_points', 'team_wins', 'steps', 'match_steps', 'env_cfg'])" or \
        #       str(obs.keys()) == "dict_keys(['units', 'units_mask', 'sensor_mask', 'map_features', 'relic_nodes', 'relic_nodes_mask', 'team_points', 'team_wins', 'delta_energy', 'steps', 'match_steps', 'env_cfg'])"



        #if obs['steps'] == 2:
        #    exit(228)


        #print('units', obs['units'].keys(), file=sys.stderr)

        #print('units position', obs['units']['position'].shape, file=sys.stderr)

        self.swap(obs['units']['position'])

        #print('units energy', obs['units']['energy'].shape, file=sys.stderr)

        self.swap(obs['units']['energy'])

        #print('units_mask', obs['units_mask'].shape, file=sys.stderr)

        self.swap(obs['units_mask'])

        #print('sensor_mask', obs['sensor_mask'].shape, file=sys.stderr)

        #if obs['steps'] == 1:
        #    print(obs['map_features']['tile_type'], file=sys.stderr)
            #time.sleep(random.randint(0, 1000000) / 1000000)

        #print('map_features', obs['map_features'].keys(), file=sys.stderr)
        #print('map_features energy', obs['map_features']['energy'].shape, file=sys.stderr)
        #print('map_features tile_type', obs['map_features']['tile_type'].shape, file=sys.stderr)


        #print('relic_nodes', obs['relic_nodes'].shape, file=sys.stderr)
        #print('relic_nodes_mask', obs['relic_nodes_mask'].shape, file=sys.stderr)
        #print('team_points', obs['team_points'], file=sys.stderr)

        self.swap(obs['team_points'])

        #print('team_wins', obs['team_wins'], file=sys.stderr)

        self.swap(obs['team_wins'])

        #print('steps', obs['steps'], file=sys.stderr)
        #print('match_steps', obs['match_steps'], file=sys.stderr)



        return obs

    def simple_act(self, step: int, obs, remainingOverageTime: int = 60):
        obs['env_cfg'] = self.env_cfg
        flipped_obs = self.flip_obs(copy.deepcopy(obs))

        by_player = {
            'player_0': obs if self.player == 'player_0' else flipped_obs,
            'player_1': obs if self.player == 'player_1' else flipped_obs,
        }

        player_id = 0 if self.player == 'player_0' else 1
        if self.prev_hidden_state_h.shape[1] != 1:
            self.prev_hidden_state_h = self.prev_hidden_state_h[:, player_id, ...].unsqueeze(1)
            self.prev_hidden_state_c = self.prev_hidden_state_c[:, player_id, ...].unsqueeze(1)
            self.prev_prediction = self.prev_prediction[:, player_id, ...].unsqueeze(1)

        env_output = self.env.reset(force=True, kaggle_observation=(by_player, player_id))
        env_output['info']['rnn_hidden_state_h_GPU'] = self.prev_hidden_state_h
        env_output['info']['rnn_hidden_state_c_GPU'] = self.prev_hidden_state_c
        env_output['info']['prediction_GPU_CPU'] = self.prev_prediction

        if obs['steps'] == 0: # just skip first step
            result = self.env.step(None)['info']
            if DEBUG__:
                self.prev_actions_taken = result['actions_taken_GPU_CPU']
            kaggle_actions = result['LOGGING_CPU_kaggle_actions'][self.player].cpu().numpy()[0]
            return kaggle_actions

        self.dbg_steps += 1
        if DEBUG__:
            env_output['info']['actions_taken_GPU_CPU'] = self.prev_actions_taken
            self.out_env_hashes(env_output, self.file, self.dbg_steps)

        with torch.no_grad():
            agent_output = self.actor_model(
                env_output, sample=False, one_player=player_id
            )
            self.prev_hidden_state_h = agent_output['rnn_hidden_state_h_GPU']
            self.prev_hidden_state_c = agent_output['rnn_hidden_state_c_GPU']
            self.prev_prediction = agent_output['prediction_GPU_CPU']

            if DEBUG__:
                self.out_env_hashes(agent_output, self.file, self.dbg_steps)

        result = self.env.step(agent_output['actions_GPU'])['info']

        if DEBUG__:
            self.prev_actions_taken = result['actions_taken_GPU_CPU']

        kaggle_actions = result['LOGGING_CPU_kaggle_actions'][self.player].cpu().numpy()[0]
        return kaggle_actions

    def data_augmentation_act(self, step: int, obs, remainingOverageTime: int = 60):
        enable_one_player_inference = os.getenv('LOCAL_RUNNER') != '1' and remainingOverageTime < MIN_OVERAGE_TIME_FOR_AUG

        if enable_one_player_inference:
            print("opi_enabled", file=sys.stderr)

        if os.getenv('LOCAL_RUNNER') == '1':
            assert enable_one_player_inference == False

        obs['env_cfg'] = self.env_cfg

        round_n = max(0, (step - 1)) // 101

        if use_weak_model:
            self.is_weak = False
            if self.current_is_weak:
                self.is_weak = True
            elif strong_all_rounds == False and round_n != 2:
                self.is_weak = True

        if not self.flip_learning:
            rotation_variants = [
                (rotation.no_rotate, 'player_0'),
                (rotation.rotate_main_diagonal, 'player_1'),
                (rotation.rotate_secondary_diagonal, 'player_0'),
                (rotation.rotate_two_diagonal, 'player_1'),
            ]
        else:
            if self.player == 'player_0':
                rotation_variants = [
                    (rotation.no_rotate, 'player_0'),
                    (rotation.rotate_secondary_diagonal, 'player_1'),
                ]
            else:
                rotation_variants = [
                    (rotation.rotate_secondary_diagonal, 'player_0'),
                    (rotation.no_rotate, 'player_1'),
                ]
        all_obs = []

        flip_obs = self.flip_obs(copy.deepcopy(obs))
        
        for rotation_variant, player in rotation_variants:
            cur_obs = obs if player == self.player else flip_obs
            rotated_obs = rotation.rotate_obs(cur_obs, rotation_variant)
            all_obs.append(rotated_obs)


        if not self.flip_learning:
            observations = [
                {
                    'player_0': all_obs[0],
                    'player_1': all_obs[1]
                },
                {
                    'player_0': all_obs[2],
                    'player_1': all_obs[3]
                }
            ]
        else:
            observations = [
                {
                    'player_0': all_obs[0],
                    'player_1': all_obs[1]
                }
            ]



        player_id = 0 if self.player == 'player_0' else 1
        if enable_one_player_inference and self.prev_hidden_state_h.shape[1] != 1:
            self.prev_hidden_state_h = self.prev_hidden_state_h[:, player_id, ...].unsqueeze(1)
            self.prev_hidden_state_c = self.prev_hidden_state_c[:, player_id, ...].unsqueeze(1)
            self.prev_prediction = self.prev_prediction[:, player_id, ...].unsqueeze(1)

        env_output = self.env.reset(force=True, kaggle_observations=((observations, player_id) if enable_one_player_inference else observations))
        env_output['info']['rnn_hidden_state_h_GPU'] = self.prev_hidden_state_h
        env_output['info']['rnn_hidden_state_c_GPU'] = self.prev_hidden_state_c
        env_output['info']['prediction_GPU_CPU'] = self.prev_prediction


        if obs['steps'] == 0: # just skip first step
            result = self.env.step(None)['info']
            kaggle_actions = result['LOGGING_CPU_kaggle_actions'][self.player].cpu().numpy()[0]
            return kaggle_actions


        with torch.no_grad():
            agent_output = self.actor_model(
                env_output, sample=False, one_player=(player_id if enable_one_player_inference else None)
            )
            self.prev_hidden_state_h = agent_output['rnn_hidden_state_h_GPU']
            self.prev_hidden_state_c = agent_output['rnn_hidden_state_c_GPU']
            self.prev_prediction = agent_output['prediction_GPU_CPU']

        sapper_logits, worker_logits = self.get_augmentation_logits(agent_output,  step, one_player=(player_id if enable_one_player_inference else None))

        actions = {}

        if not use_weak_model:
            assert not self.is_weak
        actions['worker'] = DictSeparateActorResn.logits_to_actions(worker_logits.view(-1, 6), self.is_weak, 1)
        actions['worker'] = actions['worker'].view(*worker_logits.shape[:-1], -1)
        actions['sapper'] = DictSeparateActorResn.logits_to_actions(sapper_logits.view(-1, 225), self.is_weak, 1)
        actions['sapper'] = actions['sapper'].view(*sapper_logits.shape[:-1], -1)

        result = self.env.step(actions)['info']

        if not self.flip_learning:
            # DEPRECATED
            kaggle_actions = result['LOGGING_CPU_kaggle_actions']['player_0'].cpu().numpy()[0]
        else:
            kaggle_actions = result['LOGGING_CPU_kaggle_actions'][self.player].cpu().numpy()[0]
        return kaggle_actions

    def act(self, step, obs, remainingOverageTime: int = 60):
        if self.data_augmentation:
            return self.data_augmentation_act(step, obs, remainingOverageTime)
        else:
            return self.simple_act(step, obs, remainingOverageTime)

    def get_augmentation_logits(self, agent_output, step, one_player=None):
        # sappers = torch.full((2, 1, 2, 16, 225), 1)
        # workers = torch.full((2, 1, 2, 16, 6), 1)
        sappers = agent_output['policy_logits_GPU_CPU']['sapper']
        workers = agent_output['policy_logits_GPU_CPU']['worker']

        if not self.flip_learning:
            i, j, k = (0, 0, 1, 1), (0, 0, 0, 0), (0, 1, 0, 1)
            actions = (rotation.no_rotate, rotation.rotate_main_diagonal, rotation.rotate_secondary_diagonal, rotation.rotate_two_diagonal)
            rotate_actions = (rotation.no_rotate, rotation.rotate_action_main_diagonal, rotation.rotate_action_secondary_diagonal, rotation.rotate_action_two_diagonal)
        else:
            i, j, k = (0, 0), (0, 0), (0, 1)

            if self.player == 'player_0':
                actions = (rotation.no_rotate, rotation.rotate_main_diagonal)
                rotate_actions = (rotation.no_rotate, rotation.rotate_action_main_diagonal)
            else:
                actions = (rotation.rotate_main_diagonal, rotation.no_rotate)
                rotate_actions = (rotation.rotate_action_main_diagonal, rotation.no_rotate)

        # ____ sapper _____
        # get mean between all sap matrix
        original = torch.zeros_like(sappers[0][0][0].clone().cpu())

        sum_n = 0
        for i1, j1, k1, action in zip(i, j, k, actions):
            if one_player is not None and k1 != one_player:
                continue
            if one_player is not None:
                k1 = 0
            buffer = sappers[i1][j1][k1].clone().cpu().numpy()
            for idx, one_unit in enumerate(buffer):
                buffer[idx] = action(one_unit.reshape(15, 15)).reshape(225)

            original += torch.from_numpy(buffer)
            sum_n += 1
        original = original.float() / sum_n

        if not use_weak_model:
            assert not self.is_weak
        if self.is_weak:
            T = torch.exp(torch.tensor((1. - model_power_koef) * 2.0))
            original = original / T

        # fill final matrix to tensor (2 players will have this matrix)
        augmented_sapper = torch.zeros((1, 1, 2, 16, 225))
        if not self.flip_learning:
            # DEPRECATED
            augmented_sapper[0][0][0] = original.clone()
            # костыль-заглушка 2го игрока
            augmented_sapper[0][0][1] = original.clone()
        else:
            if False:
                if self.player == 'player_0':
                    augmented_sapper[0][0][0] = original.clone()
                    augmented_sapper[0][0][1] = agent_output['policy_logits_GPU_CPU']['sapper'][0][0][1].clone()
                else:
                    augmented_sapper[0][0][0] = agent_output['policy_logits_GPU_CPU']['sapper'][0][0][0].clone()
                    augmented_sapper[0][0][1] = original.clone()

            if True:
                if self.player == 'player_0':
                    augmented_sapper[0][0][0] = original.clone()
                    buffer = original.clone()
                    for idx, one_unit in enumerate(buffer):
                        buffer[idx] = actions[1](one_unit.reshape(15, 15)).reshape(225)

                    augmented_sapper[0][0][1] = buffer
                else:
                    buffer = original.clone()
                    for idx, one_unit in enumerate(buffer):
                        buffer[idx] = actions[0](one_unit.reshape(15, 15)).reshape(225)

                    augmented_sapper[0][0][0] = buffer
                    augmented_sapper[0][0][1] = original.clone()

        # ____ worker _____
        original = torch.zeros_like(workers[0][0][0].clone().cpu())

        sum_n = 0
        for i1, j1, k1, rotate_action in zip(i, j, k, rotate_actions):
            if one_player is not None and k1 != one_player:
                continue
            if one_player is not None:
                k1 = 0

            buffer = workers[i1][j1][k1].clone().cpu().numpy()
            for idx, one_unit in enumerate(buffer):
                rotated = np.zeros((6,), dtype=np.float32)
                for old_action, value in enumerate(one_unit):
                    new_action = rotate_action(old_action)
                    rotated[new_action] = float(value)
                buffer[idx] = rotated


            original += torch.from_numpy(buffer)
            sum_n += 1
        original = original.float() / sum_n


        if not use_weak_model:
            assert not self.is_weak
        if self.is_weak:
            T = torch.exp(torch.tensor((1. - model_power_koef) * 2.0))
            original = original / T


        augmented_worker = torch.zeros((1, 1, 2, 16, 6))
        if not self.flip_learning:
            # DEPRECATED
            augmented_worker[0][0][0] = original.clone()
            # костыль-заглушка 2го игрока
            augmented_worker[0][0][1] = original.clone()
        else:
            if False:
                if self.player == 'player_0':
                    augmented_worker[0][0][0] = original.clone()
                    augmented_worker[0][0][1] = agent_output['policy_logits_GPU_CPU']['worker'][0][0][1].clone()
                else:
                    augmented_worker[0][0][0] = agent_output['policy_logits_GPU_CPU']['worker'][0][0][0].clone()
                    augmented_worker[0][0][1] = original.clone()

            if True:
                if self.player == 'player_0':
                    augmented_worker[0][0][0] = original.clone()

                    buffer = original.clone().cpu().numpy()
                    for idx, one_unit in enumerate(buffer):
                        rotated = np.zeros((6,), dtype=np.float32)
                        for old_action, value in enumerate(one_unit):
                            new_action = rotate_actions[1](old_action)
                            rotated[new_action] = float(value)
                        buffer[idx] = rotated

                    augmented_worker[0][0][1] = torch.from_numpy(buffer)
                else:

                    buffer = original.clone().cpu().numpy()
                    for idx, one_unit in enumerate(buffer):
                        rotated = np.zeros((6,), dtype=np.float32)
                        for old_action, value in enumerate(one_unit):
                            new_action = rotate_actions[0](old_action)
                            rotated[new_action] = float(value)
                        buffer[idx] = rotated


                    augmented_worker[0][0][0] = torch.from_numpy(buffer)
                    augmented_worker[0][0][1] = original.clone()

        return augmented_sapper, augmented_worker
