# procedural-generalization

testing the generalization properties of the openai procgen benchmark



setup train-procgen environments using lines below

```
git clone https://github.com/openai/train-procgen.git
conda env update --name train-procgen --file train-procgen/environment.yml
conda activate train-procgen
pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip
pip install -e train-procgen
```

now "python test.py" will run
some comments are inside

