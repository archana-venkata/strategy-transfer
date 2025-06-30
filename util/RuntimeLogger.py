
from time import time
import json
import os


class RuntimeLogger:
    def __init__(self, log_dir: str):
        print(log_dir)
        self.runtime_filepath = os.path.join(log_dir, "time_log.json")
        os.makedirs(os.path.dirname(self.runtime_filepath), exist_ok=True)
        self.runtime_dict = {}
        self.global_start_time = None

    def start(self):
        self.global_start_time = time()

    def log_task_end(self, task_name: str, task_start_time: float):
        task_runtime = time() - task_start_time
        self.runtime_dict.update({task_name: task_runtime})

    def log_experiment_end(self):
        self.log_task_end('global_runtime', self.global_start_time)
        json.dump(self.runtime_dict,
                  open(self.runtime_filepath, 'w'), indent=4)
