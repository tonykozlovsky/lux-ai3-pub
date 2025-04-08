import copy

def get_input_buffers_for_inference(env_output):
    return {
        "obs": env_output["obs"],
        "info": {
            "GPU1_available_actions_mask": env_output["info"]["GPU1_available_actions_mask"],
            "GPU1_units_masks": env_output["info"]["GPU1_units_masks"],
            "rnn_hidden_state_h_GPU": env_output["info"]["rnn_hidden_state_h_GPU"],
            "rnn_hidden_state_c_GPU": env_output["info"]["rnn_hidden_state_c_GPU"],
            "prediction_GPU_CPU": env_output["info"]["prediction_GPU_CPU"],
        },
        "done_GPU_CPU": env_output["done_GPU_CPU"]
    }


def get_output_buffers_for_inference(agent_output):
    agent_output = copy.copy(agent_output)
    agent_output.pop('teacher_output')
    return agent_output


