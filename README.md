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

To run sequence of five tasks, run the file:
```bash
python run_phase2.py
```

Tasks can be modified in `legion_marl/tasks/task_config.py`.