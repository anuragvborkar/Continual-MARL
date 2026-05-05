## Note: This repository is cloned from [facebookresearch/BenchMARL](https://github.com/facebookresearch/BenchMARL) to adapt to continual learning settings.

### Setup for local development

```bash
conda create -n benchmarl python=3.10
conda activate benchmarl
```

```bash
git clone https://github.com/anuragvborkar/Continual-MARL.git
cd Continual-MARL
pip install -e .
```
### Install environments and dependencies

```bash
pip install torchrl vmas
```

### Run

Experiments are launched with a [default configuration](benchmarl/conf) that 
can be overridden in many ways. 

#### Command line

To launch an experiment from the command line you can do

```bash
python benchmarl/run.py algorithm=mappo task=vmas/balance
```

Thanks to [hydra](https://hydra.cc/docs/intro/), you can run benchmarks as multi-runs like:
```bash
python benchmarl/run.py -m algorithm=mappo,qmix,masac task=vmas/balance,vmas/sampling seed=0,1
```
The default implementation for hydra multi-runs is sequential, but [parallel](https://hydra.cc/docs/plugins/joblib_launcher/)
 and [slurm](https://hydra.cc/docs/plugins/submitit_launcher/) launchers are also available.

#### Script

You can also load and launch your experiments from within a script

```python
 experiment = Experiment(
    task=VmasTask.BALANCE.get_from_yaml(),
    algorithm_config=MappoConfig.get_from_yaml(),
    model_config=MlpConfig.get_from_yaml(),
    critic_model_config=MlpConfig.get_from_yaml(),
    seed=0,
    config=ExperimentConfig.get_from_yaml(),
)
experiment.run()
```

You can also run multiple experiments in a `Benchmark`.

```python
benchmark = Benchmark(
    algorithm_configs=[
        MappoConfig.get_from_yaml(),
        QmixConfig.get_from_yaml(),
        MasacConfig.get_from_yaml(),
    ],
    tasks=[
        VmasTask.BALANCE.get_from_yaml(),
        VmasTask.SAMPLING.get_from_yaml(),
    ],
    seeds={0, 1},
    experiment_config=ExperimentConfig.get_from_yaml(),
    model_config=MlpConfig.get_from_yaml(),
    critic_model_config=MlpConfig.get_from_yaml(),
)
benchmark.run_sequential()
```


## Extending
One of the core tenets of BenchMARL is allowing users to leverage the existing algorithm
and tasks implementations to benchmark their newly proposed solution.

For this reason we expose standard interfaces with simple abstract methods
for [algorithms](benchmarl/algorithms/common.py), [tasks](benchmarl/environments/common.py) and [models](benchmarl/models/common.py).
To introduce your solution in the library, you just need to implement the abstract methods
exposed by these base classes which use objects from the [TorchRL](https://github.com/pytorch/rl) library.

Here is an example on how you can create a custom algorithm [![Example](https://img.shields.io/badge/Example-blue.svg)](examples/extending/algorithm).

Here is an example on how you can create a custom task [![Example](https://img.shields.io/badge/Example-blue.svg)](examples/extending/task).

Here is an example on how you can create a custom model [![Example](https://img.shields.io/badge/Example-blue.svg)](examples/extending/model).


## Configuring
As highlighted in the [run](#run) section, the project can be configured either
in the script itself or via [hydra](https://hydra.cc/docs/intro/). 
We suggest to read the hydra documentation
to get familiar with all its functionalities. 

Each component in the project has a corresponding yaml configuration in the BenchMARL 
[conf tree](benchmarl/conf). 
Components' configurations are loaded from these files into python dataclasses that act 
as schemas for validation of parameter names and types. That way we keep the best of 
both words: separation of all configuration from code and strong typing for validation! 
You can also directly load and validate configuration yaml files without using hydra from a script by calling 
`ComponentConfig.get_from_yaml()`.

### Experiment

Experiment configurations are in [`benchmarl/conf/config.yaml`](benchmarl/conf/config.yaml).
Running custom experiments is extremely simplified by the [Hydra](https://hydra.cc/) configurations.
The default configuration for the library is contained in the [`benchmarl/conf`](benchmarl/conf) folder.

When running an experiment you can override its hyperparameters like so
```bash
python benchmarl/run.py task=vmas/balance algorithm=mappo experiment.lr=0.03 experiment.evaluation=true experiment.train_device="cpu"
```

Experiment hyperparameters are loaded from [`benchmarl/conf/experiment/base_experiment.yaml`](benchmarl/conf/experiment/base_experiment.yaml)
into a dataclass [`ExperimentConfig`](benchmarl/experiment/experiment.py) defining their domain.
This makes it so that all and only the parameters expected are loaded with the right types.
You can also directly load them from a script by calling `ExperimentConfig.get_from_yaml()`.

Here is an example of overriding experiment hyperparameters from hydra 
[![Example](https://img.shields.io/badge/Example-blue.svg)](examples/configuring/configuring_experiment.sh) or from
a script [![Example](https://img.shields.io/badge/Example-blue.svg)](examples/configuring/configuring_experiment.py).

### Algorithm

You can override an algorithm configuration when launching BenchMARL.

```bash
python benchmarl/run.py task=vmas/balance algorithm=masac algorithm.num_qvalue_nets=3 algorithm.target_entropy=auto algorithm.share_param_critic=true
```

Available algorithms and their default configs can be found at [`benchmarl/conf/algorithm`](benchmarl/conf/algorithm).
They are loaded into a dataclass [`AlgorithmConfig`](benchmarl/algorithms/common.py), present for each algorithm, defining their domain.
This makes it so that all and only the parameters expected are loaded with the right types.
You can also directly load them from a script by calling `YourAlgorithmConfig.get_from_yaml()`.

Here is an example of overriding algorithm hyperparameters from hydra 
[![Example](https://img.shields.io/badge/Example-blue.svg)](examples/configuring/configuring_algorithm.sh) or from
a script [![Example](https://img.shields.io/badge/Example-blue.svg)](examples/configuring/configuring_algorithm.py).


### Task

You can override a task configuration when launching BenchMARL.
However this is not recommended for benchmarking as tasks should have fixed version and parameters for reproducibility.

```bash
python benchmarl/run.py task=vmas/balance algorithm=mappo task.n_agents=4
```

Available tasks and their default configs can be found at [`benchmarl/conf/task`](benchmarl/conf/task).
They are loaded into a dataclass [`TaskConfig`](benchmarl/environments/common.py), defining their domain.
Tasks are enumerations under the environment name. For example, `VmasTask.NAVIGATION` represents the navigation task in the
VMAS simulator. This allows autocompletion and seeing all available tasks at once.
You can also directly load them from a script by calling `YourEnvTask.TASK_NAME.get_from_yaml()`.

Here is an example of overriding task hyperparameters from hydra 
[![Example](https://img.shields.io/badge/Example-blue.svg)](examples/configuring/configuring_task.sh) or from
a script [![Example](https://img.shields.io/badge/Example-blue.svg)](examples/configuring/configuring_task.py).

### Model

You can override the model configuration when launching BenchMARL.
By default an MLP model will be loaded with the default config.
You can change it like so:

```bash
python benchmarl/run.py task=vmas/balance algorithm=mappo model=layers/mlp model=layers/mlp model.layer_class="torch.nn.Linear" "model.num_cells=[32,32]" model.activation_class="torch.nn.ReLU"
```

Available models and their configs can be found at [`benchmarl/conf/model/layers`](benchmarl/conf/model/layers).
They are loaded into a dataclass [`ModelConfig`](benchmarl/models/common.py), defining their domain.
You can also directly load them from a script by calling `YourModelConfig.get_from_yaml()`.

Here is an example of overriding model hyperparameters from hydra 
[![Example](https://img.shields.io/badge/Example-blue.svg)](examples/configuring/configuring_model.sh) or from
a script [![Example](https://img.shields.io/badge/Example-blue.svg)](examples/configuring/configuring_model.py).

#### Sequence model
You can compose layers into a sequence model.
Available layer names are in the [`benchmarl/conf/model/layers`](benchmarl/conf/model/layers) folder.

```bash
python benchmarl/run.py task=vmas/balance algorithm=mappo model=sequence "model.intermediate_sizes=[256]" "model/layers@model.layers.l1=mlp" "model/layers@model.layers.l2=mlp" "+model/layers@model.layers.l3=mlp" "model.layers.l3.num_cells=[3]"
```
Add a layer with `"+model/layers@model.layers.l3=mlp"`.

Remove a layer with `"~model.layers.l2"`.

Configure a layer with `"model.layers.l1.num_cells=[3]"`.

Here is an example of creating a sequence model from hydra 
[![Example](https://img.shields.io/badge/Example-blue.svg)](examples/configuring/configuring_sequence_model.sh) or from
a script [![Example](https://img.shields.io/badge/Example-blue.svg)](examples/configuring/configuring_sequence_model.py).

## Features

BenchMARL has several features:
- A test CI with integration and training test routines that are run for all simulators and algorithms
- Integration in the official TorchRL ecosystem for dedicated support
- Possibility of using different algorithms and models for different agent groups (see [`examples/ensemble`](examples/ensemble))


### Logging

BenchMARL is compatible with the [TorchRL loggers](https://github.com/pytorch/rl/tree/main/torchrl/record/loggers).
A list of logger names can be provided in the [experiment config])(benchmarl/conf/experiment/base_experiment.yaml.
Example of available options are: `wandb`, `csv`, `mflow`, `tensorboard` or any other option available in TorchRL. You can specify the loggers
in the yaml config files or in the script arguments like so:
```bash
python benchmarl/run.py algorithm=mappo task=vmas/balance "experiment.loggers=[wandb]"
```
The wandb logger is fully compatible with experiment restoring and will automatically resume the run of 
the loaded experiment.

### Checkpointing

Experiments can be checkpointed every `experiment.checkpoint_interval` collected frames.
Experiments will use an output folder for logging and checkpointing which can be specified in `experiment.save_folder`.
If this is left unspecified,
the default will be the hydra output folder (if using hydra) or (otherwise) the current directory 
where the script is launched.
The output folder will contain a folder for each experiment with the corresponding experiment name.
Their checkpoints will be stored in a `"checkpoints"` folder within the experiment folder.
```bash
python benchmarl/run.py task=vmas/balance algorithm=mappo experiment.max_n_iters=3 experiment.on_policy_collected_frames_per_batch=100 experiment.checkpoint_interval=100
```

To load from a checkpoint, pass the absolute checkpoint file name to `experiment.restore_file`.
```bash
python benchmarl/run.py task=vmas/balance algorithm=mappo experiment.max_n_iters=6 experiment.on_policy_collected_frames_per_batch=100 experiment.restore_file="/hydra/experiment/folder/checkpoint/checkpoint_300.pt"
```
Here is a python example when modifying the config
[![Example](https://img.shields.io/badge/Example-blue.svg)](examples/checkpointing/reload_experiment.py)
and one keeping the same config 
[![Example](https://img.shields.io/badge/Example-blue.svg)](examples/checkpointing/resume_experiment.py).

There are also ways to **resume** and **evaluate** hydra experiments directly from the file
```bash
python benchmarl/evaluate.py ../outputs/2024-09-09/20-39-31/mappo_balance_mlp__cd977b69_24_09_09-20_39_31/checkpoints/checkpoint_100.pt
```
```bash
python benchmarl/resume.py ../outputs/2024-09-09/20-39-31/mappo_balance_mlp__cd977b69_24_09_09-20_39_31/checkpoints/checkpoint_100.pt
```

### Callbacks

Experiments optionally take a list of [`Callback`](benchmarl/experiment/callback.py) which have several methods
that you can implement to see what's going on during training such 
as `on_batch_collected`, `on_train_end`, and `on_evaluation_end`.

[![Example](https://img.shields.io/badge/Example-blue.svg)](examples/callback/custom_callback.py)


## Citing BenchMARL

If you use BenchMARL in your research please use the following BibTeX entry:

```BibTeX
@article{bettini2024benchmarl,
  author  = {Matteo Bettini and Amanda Prorok and Vincent Moens},
  title   = {BenchMARL: Benchmarking Multi-Agent Reinforcement Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {217},
  pages   = {1--10},
  url     = {http://jmlr.org/papers/v25/23-1612.html}
}
```

## License
BenchMARL is licensed under the MIT License. See [LICENSE](LICENSE) for details.
