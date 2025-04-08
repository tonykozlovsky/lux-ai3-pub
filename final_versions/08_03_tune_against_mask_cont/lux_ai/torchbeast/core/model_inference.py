import copy
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import torch
import setproctitle
import logging

from .buffer_utils import fill_buffers_inplace, buffers_apply

class ModelInferenceProcess:
    def __init__(self, actor_models, frozen_actor_models, frozen_teacher_actors, inference_input_buffers, inference_output_buffers, flags, num_processes=1):
        """
        Initialize inference processes for running model predictions.

        Args:
            actor_model: The model to run inference with (already on correct device)
            num_processes: Number of inference processes to maintain
        """
        self.input_queues = []
        self.output_queues = []
        self.processes = []
        self.num_processes = num_processes
        self.free_process_queue = mp.Queue()
        for i in range(num_processes):
            self.free_process_queue.put(i)

        self.free_buffer_queue = mp.Queue()
        for i in range(len(inference_input_buffers)):
            self.free_buffer_queue.put(i)

        assert len(inference_input_buffers) == len(inference_output_buffers)

        self.input_buffers = inference_input_buffers
        self.output_buffers = inference_output_buffers

        self.iteration = 0

        for i in range(num_processes):
            input_queue = mp.Queue()
            output_queue = mp.Queue()

            self.input_queues.append(input_queue)
            self.output_queues.append(output_queue)

        def start_process(i):
            cur_dev_id = i % flags.n_actor_devices
            inference_device = cur_dev_id
            process = mp.Process(
                target=self._inference_worker,
                args=(actor_models[cur_dev_id], frozen_actor_models[cur_dev_id], frozen_teacher_actors[cur_dev_id], i, self.input_buffers, self.output_buffers, inference_device, self.input_queues[i], self.output_queues[i])
            )
            process.start()
            return process

        with ThreadPoolExecutor(max_workers=min(32, num_processes)) as executor:
            self.processes = list(executor.map(start_process, range(num_processes)))

    def __getstate__(self):
        # Do not pickle processes
        return {
            'input_queues': self.input_queues,
            'output_queues': self.output_queues,
            'num_processes': self.num_processes,
            'free_process_queue': self.free_process_queue,
            'input_buffers': self.input_buffers,
            'output_buffers': self.output_buffers,
            'free_buffer_queue': self.free_buffer_queue,
            'iteration': self.iteration
        }

    def __setstate__(self, state):
        self.input_queues = state['input_queues']
        self.output_queues = state['output_queues']
        self.num_processes = state['num_processes']
        self.free_process_queue = state['free_process_queue']
        self.input_buffers = state['input_buffers']
        self.output_buffers = state['output_buffers']
        self.free_buffer_queue = state['free_buffer_queue']
        self.iteration = state['iteration']
        # Initialize empty processes list since processes can't be pickled
        self.processes = []


    # Currently in inference worker receives all frozen models, but it might be improved later
    @staticmethod
    def _inference_worker(actor_model, frozen_actor_models, frozen_teacher_model, id, input_buffers, output_buffers, inference_device, input_queue, output_queue):
        """Worker process that handles model inference"""

        actor_model.eval()
        for frozen_model_actor in frozen_actor_models:
            frozen_model_actor.eval()
        if frozen_teacher_model is not None:
            frozen_teacher_model.eval()

        assert len(input_buffers) == len(output_buffers)

        logging.info(f"Inference worker {id} started.")

        setproctitle.setproctitle(f"INFERENCE_{id}_PROCESS")


        while True:
            try:
                # Get input from queue
                buffer_id, frozen_model_id, is_frozen_teacher, kwargs = input_queue.get()
                pinned_buffers = buffers_apply(input_buffers[buffer_id], lambda x: x.pin_memory())
                env_output = buffers_apply(pinned_buffers, lambda x: x.to(inference_device, non_blocking=True))

                #print(buffer_id, frozen_model_id, is_frozen_teacher, kwargs)

                with torch.no_grad():
                    if is_frozen_teacher:
                        output = frozen_teacher_model(env_output, **kwargs)
                    elif frozen_model_id is not None:
                        assert frozen_model_id < len(frozen_actor_models)
                        output = frozen_actor_models[frozen_model_id](env_output, **kwargs)
                    else:
                        output = actor_model(env_output, **kwargs)


                fill_buffers_inplace(output_buffers[buffer_id], output)

                output_queue.put(42)
            except Exception as e:
                print(f"Error in inference worker: {e} ")
                output_queue.put(e)

    def infer(self, profiler, env_output, frozen_model_id=None, is_frozen_teacher=False, **kwargs):
        """
        Run inference on the model in a separate process.

        Args:
            env_output: Environment output to process
            **kwargs: Additional arguments to pass to the model

        Returns:
            Model output
        """
        buffer_id = self.free_buffer_queue.get()
        fill_buffers_inplace(self.input_buffers[buffer_id], env_output)

        # Get an available worker
        self.iteration += 1
        if self.iteration % 200 == 0:
            logging.info(f"Free inference workers: {self.free_process_queue.qsize()}")
        process_idx = self.free_process_queue.get()
        input_queue = self.input_queues[process_idx]
        output_queue = self.output_queues[process_idx]
        try:
            # Use the worker
            input_queue.put((buffer_id, frozen_model_id, is_frozen_teacher, kwargs))
            result = output_queue.get()
            if isinstance(result, Exception):
                print(f"Error in inference worker: {result}")
                raise result
            with profiler.block("result_clone"):
                result = buffers_apply(self.output_buffers[buffer_id], lambda x: x.clone())
            return result
        finally:
            # Return worker to pool
            self.free_process_queue.put(process_idx)
            self.free_buffer_queue.put(buffer_id)

    def __del__(self):
        """Cleanup processes on deletion"""
        for process in self.processes:
            process.terminate()
            process.join()
