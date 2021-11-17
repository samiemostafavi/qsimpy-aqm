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

2. Move into the repository, create a Python 3.6.0 virtual environment (requires [virtualenv](https://pypi.org/project/virtualenv/)), and activate it:

    ``` bash
    $ cd time-sensitive-aqm

    $ python -m virtualenv --python=python3.6.0 ./venv

    $ source ./venv/bin/activate

    (venv) $ 
    ```

3. Install the required packages by `requirements.txt`: `pip install -Ur requirements.txt`.
