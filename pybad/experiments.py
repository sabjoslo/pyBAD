"""Task-agnostic infrastructure for simulating sequential experiments.
"""

import logging
import numpy as np
import os
import pickle
from tqdm import tqdm
from pybad.bad.sbinom import update_models

###### Helper functions for automatically saving progress during the experiment ######

def make_filepath(filename, path):
    return f"{path}/{filename}"

def load_from_file(filepath):
    if os.path.exists(filepath):
        return pickle.load(open(filepath, "rb"))
    return []

def save_to_file(obj, filepath):
    with open(filepath, "wb") as wfh:
        pickle.dump(obj, wfh)

def record(data, path):
    if isinstance(path, type(None)):
        return
    for k, v in data.items():
        save_to_file(v, make_filepath(k, path))

def setup(path, *models):
    if not isinstance(path, type(None)):
        if not os.path.exists(path):
            os.mkdir(path)
        models_ = load_from_file(make_filepath("models", path))
        pm = load_from_file(make_filepath("modelprobs", path))
        samples = load_from_file(make_filepath("samples", path))
        W = load_from_file(make_filepath("weights", path))
        X = load_from_file(make_filepath("X", path))
        Y = load_from_file(make_filepath("Y", path))
        t = len(samples)-1
        if t == -1:
            assert ( 
                len(models_) == len(pm) == len(samples) == len(W) == len(X) == \
                len(Y)
            )
            models_ = models
            pm.append([ m.p_m for m in models ])
            samples.append([ m.dist.samples for m in models ])
            W.append([ m.dist.W for m in models ])
        elif t > -1:
            assert len(pm) == len(samples) == len(W) == len(X)+1 == len(Y)+1
            assert ( 
                len(models) == len(models_) == len(pm[0]) == len(samples[0]) == \
                len(W[0])
            )
        logging.info(f"Left off at trial {t}")
    else:
        models_ = models
        pm = [[ m.p_m for m in models ]]
        samples = [[ m.dist.samples for m in models ]]
        W = [[ m.dist.W for m in models ]]
        X = []
        Y = []
        t = -1
    return t, models_, dict(
        models=models_, modelprobs=pm, samples=samples, weights=W, X=X, Y=Y
    )

def run(
    design_method, response_function, models, ntrials=100, path=None, 
    update=update_models
):
    """Run a simulated experiment.
    
    Parameters
    ----------
    design_method : callable
        The sequential design method.
    response_function : callable
        Function simulating behavioral responses.
    models : array_like
        Set of candidate models. If `path` is specified and model objects are
        already written to disk at the indicated path, these are overwritten by
        the model objects written to disk.
    ntrials : int, optional
        Number of trials in the experiment, default is 100.
    path : string, optional
        Path to save data from the experiment, default is None, in which case
        data is not saved to disk.
    update : callable, optional
        Function for updating models on the basis of observed data, default is
        `pybad.bad.sbinom.update_models`.
        
    Returns
    -------
    data : dict
        Returns a dictionary with keys "X" (designs selected during the 
        experiment), "Y" (responses observed during the experiment),
        "modelprobs" (probabilities corresponding to each model in `models`), 
        "samples" (importance samples drawn on each trial from each model in 
        `models`), "weights" (importance weights assigned on each trial from 
        each model in `models`), and "models" (the set of updated `pybad.Model` 
        objects, which can be used to continue an interrupted experiment).
        
    Examples
    --------
    Simulate an experiment to recover the power-law model of memory retention
    using adaptive design optimization:
    
    >>> from pybad.experiments import run
    >>> from pybad.memory_retention import *
    >>> run(design_method=ado, response_function=response_function(
    >>>     true_model=pow_f, true_params=(.9025,.4861)
    >>> ), models=init_models(), ntrials=10, update=update_models)
    """
    t, models, data = setup(path, *models)
    for ti in tqdm(range(ntrials)):
        if t > ti:
            continue
        if t == ti:
            logging.info(f"Imputing saved data from trial {ti}")
            for mi, m in enumerate(models):
                assert np.array_equal(m.dist.samples, data["samples"][ti][mi])
                assert np.array_equal(m.dist.W, data["weights"][ti][mi])
                assert np.equal(m.p_m, data["modelprobs"][ti][mi])
        logging.info(f"At trial {ti}.")
        x = design_method(ti, ntrials-ti-1, *models)
        y = response_function(x)
        update(y, x, *models)
        data["X"].append(x)
        data["Y"].append(y)
        data["modelprobs"].append([ m.p_m for m in models ])
        data["samples"].append([ m.dist.samples for m in models ])
        data["weights"].append([ m.dist.W for m in models ])
        data["models"] = models
        record(data, path)
        
    return data