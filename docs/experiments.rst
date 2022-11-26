Help on module experiments:

NAME
    experiments - Task-agnostic infrastructure for simulating sequential experiments.

FUNCTIONS
    load_from_file(filepath)

    make_filepath(filename, path)

    record(data, path)

    run(design_method, response_function, models, ntrials=100, path=None, update=<function update_models at 0x104cbcd30>)
        Run a simulated experiment.

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

    save_to_file(obj, filepath)

    setup(path, *models)
