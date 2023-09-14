# Materials for Substra workshop


This repository contains all materials to follow along the Substra hands-on workshop.

## How to run the workshop locally

First, clone this repository.

```
git clone git@github.com:Substra/substra-workshops.git
```
Depending on your preferences, you can either install the requirements in a virtual env or in a conda env.

* Install requirements with virtual env and pip

This workshop is compatible with Python 3.8, 3.9 and 3.10.

    ```
    cd substra-workshops
    python3 -m venv substra-venv
    source substra-venv/bin/activate
    pip install -r requirements.txt
    ```

* Install requirements with conda

    ```
    cd substra-workshops
    conda create -n substra python=3.10 pip
    conda activate substra
    pip install -r requirements.txt
    ```

Then launch the notebook.

```
jupyter notebook
```


## How to get help

If you are running this workshop on your own and have any questions, feel free to reach us on [Slack](https://join.slack.com/t/substra-workspace/shared_invite/zt-1fqnk0nw6-xoPwuLJ8dAPXThfyldX8yA).
