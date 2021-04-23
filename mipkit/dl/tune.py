# Copyright (c) 2021 congvm
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
from ray import tune
from ray.tune import CLIReporter, logger
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from tensorboard import program


def run_tensorboard(log_dir, host, port):
    """Run tensorboard as a daemon process

    Args:
        log_dir (str): log directory.
        host (str): tensorboard host 
        port (str): tensorboard port 

    Returns:
        str: url of tensorboard server
    """

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir,
                       '--port', port, '--host', host])
    url = tb.launch()
    print(f'Tensorboard started at {url}')
    return url


def start_tuning(train_config: dict,
                 tuning_config: dict,
                 training_func,
                 experiment_name: str,
                 report_metric_columns: list,
                 monitor_metric: str,
                 monitor_mode: str,
                 num_epochs: int = 40,
                 num_samples: int = 10,
                 gpus_per_trial: int = 1,
                 cpus_per_trial: int = 1,
                 log_dir: str = '.',
                 tb_port: str = '6006',
                 tb_host: str = '0.0.0.0',
                 run_tb: bool = False):
    # ===============================================================================
    # Configuration
    # ===============================================================================

    log_path = os.path.join(log_dir, experiment_name)

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    # scheduler = PopulationBasedTraining(
    #     time_attr="training_iteration",
    #     metric=monitor_metric,
    #     mode=monitor_mode,
    #     perturbation_interval=4,
    #     hyperparam_mutations={
    #         "lr": lambda: tune.loguniform(1e-4, 1e-1).func(None),
    #         "batch_size": [32, 64, 128]
    #     })

    reporter = CLIReporter(
        parameter_columns=list(tuning_config.keys()),
        metric_columns=report_metric_columns)

    # ===============================================================================
    # Tensorboard
    # ===============================================================================
    if run_tb:
        run_tensorboard(log_path, host=tb_host, port=tb_port)

    # ===============================================================================
    # Tune Run
    # ===============================================================================
    analysis = tune.run(
        tune.with_parameters(
            training_func, **train_config),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric=monitor_metric,
        mode=monitor_mode,
        config=tuning_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=experiment_name,
        verbose=1,
        local_dir=log_dir)

    print("Best hyperparameters found were: ", analysis.best_config)
    analysis.results_df.to_csv(os.path.join(log_path, 'results.csv'),
                               index=False)
    analysis.best_result_df.to_csv(os.path.join(log_path, 'best_results.csv'),
                                   index=False)
