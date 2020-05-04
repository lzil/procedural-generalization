# procedural-generalization

testing the generalization properties of the openai procgen benchmark


## setup

setup train-procgen environments using lines below

```
git clone https://github.com/openai/train-procgen.git
conda env update --name train-procgen --file train-procgen/environment.yml
conda activate train-procgen
pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip  
pip install pyyaml
pip install -e train-procgen
```


---

## training a baseline agent

Simply run `train.py` in `baseline_agent/` with appropriate parameters.

Parameters can be entered with a config file with a `-c` flag.
Example config files in `baseline_agent/configs/`.

The default parameters are a good place to start; check out the `train-procgen` repo for more details.

Sample run:
`python train.py -c test.yaml`


---

## training with t-rex

T-REX consists of 4 main steps.

1. train an agent with PPO to do well on an environment, in general.
    - any existing PPO trainer will do, for instance the one in `baseline_agent/train.py`
    - save checkpoints of these models into some directory. example: `chaser_model_dir`
2. sample trajectories from those trained models, from a minimal number of levels, and sort according to rewards obtained
    - done in `reward_model.py`
    - demonstrations/trajectories generated in `generate_procgen_demonstrations`
3. use those trajectories to train a reward model
    - done in `reward_model.py`
    - first create the training data `create_training_data`, then train a `RewardNet` with a `RewardTrainer`
4. train another agent with that reward model
    - done in `train_policy.py`
    - load the reward model saved from step 3 to train an actual policy


TODO: sample runs, config files

---

## calculating and plotting correlations

We can evaluate the quality of a given model with a simple metric: the correlation between the real return of a trajectory and the predicted return from the reward model.

The code for evaluating the correlation is in `reward_metric.py`, and the wrapper code that does all this in an organized fashion is in `plot_correlations.py`.

Simply run `plot_correlations.py` with the right parameters in the main function.
Initially, set 'corrs_from_file' to False; after each run that is run this way, the program will run `calc_correlations` (which take time to run) will be saved in a JSON file.
Afterwards, the program will attempt to plot the correlations with `plot_correlations`; if 'corrs_from_file' is True, then the cached correlations file will be used.


---

## recording and watching videos of saved models

Tesing a trained agent saved in `policy.parameters` can be done with e.g.:  
`python test_policy.py --load_path policy.parameters --env_name Name`

You can also record a video of the trained agent with e.g.:  
`python rec_video.py --load_path policy.parameters --env_name Name`

Additional argumets are available



