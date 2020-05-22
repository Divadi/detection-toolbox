from comet_ml import Experiment
import time
import os

'''
This file sends the contents of a log file (that some other process constantly updates with more info), 
and uploads it to comet output.
Inputs:
    project_name: Project name on comet 
    exp_name: Experiment name on comet
    log_file_path: Log file to keep track of/send
    refresh_rate: Time between checks of log_file to see if anything changed. (seconds)
'''
def log_file_to_comet_output(
    project_name,
    exp_name,
    log_file_path,
    refresh_rate
):
    experiment = Experiment(
        api_key = os.environ["COMET_API_KEY"],
        project_name = project_name
    )
    experiment.set_name(exp_name)

    with open(log_file_path, "r") as f:
        while True:
            print(f.read(), end='')
            time.sleep(refresh_rate)