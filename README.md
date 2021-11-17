# time-sensitive-aqm

## Setting up

1. Clone this repository.
   Note that we use submodules to link against other related repositories, and the cloning process is different.
   The easiest way of doing it is by adding the `--recurse-submodules` flag when initially cloning: `git clone --recurse-submodules git@github.com:samiemostafavi/time-sensitive-aqm.git`.
   Alternatively, if by accident you cloned this repository as you normally would, you can initialize the submodules from inside the repo itself:

   ```bash
   # we clone the repository
   $ git clone git@github.com:samiemostafavi/time-sensitive-aqm.git
   ...

   # afterwards, move into the repository and initialize the submodules
   $ cd time-sensitive-aqm

   $ git submodule init

   $ git submodule update

   ```

2. Move into the `conditional-latency-probability-prediction` repository, create a Python 3.6.0 virtual environment (requires [virtualenv](https://pypi.org/project/virtualenv/)), and activate it:

    ``` bash
    $ cd time-sensitive-aqm/conditional-latency-probability-prediction

    $ python -m virtualenv --python=python3.6.0 ./venv

    $ source ./venv/bin/activate

    (venv) $ 
    ```

3. Install the required packages by `requirements.txt`: `pip install -Ur requirements.txt`.

### Working with submodules

Linked submodules are "frozen" on a specific commit, not a branch as you might otherwise expect.
In case you made changes to one submodule and then wanted to link the new state to its own and this repository, the procedure is the following:

1. Move into the submodule directory.
2. Add, commit, and push your changes to upstream: `git add . && git commit -m <commit comment> && git push`.
3. Move back out into the main directory of this repository.
4. Add, commit, and push the modified submodule: `git add <submodule dir> && git commit -m <updated submodule X> && git push`.

In case the submodule was updated separately and you wish to add those changes to this project:

1. Move into the submodule directory.
2. Pull the changes: `git pull`.
3. Move back out into the main directory of this repository.
4. Add, commit, and push the modified submodule: `git add <submodule dir> && git commit -m <updated submodule X> && git push`.


