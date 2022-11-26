Help on module sbinom:

NAME
    sbinom

DESCRIPTION
    Implements utility functions and distribution update functions for sequential
    binomial trials, i.e., for binary class models for which a single response
    y ~ Bernoulli(f(x)) is collected on each trial.

FUNCTIONS
    U(designs, u, *models)
        Calculate global utility for a set of candidate designs.

        Parameters
        ----------
        design_grid : (# of candidate designs, # of design attributes)-array
            Set of candidate designs.
        u : callable
            The local utility function. One of {`u_parameterEstimaton`,
            `u_modelSelection`, `u_totalEntropy`}, or a user-defined function. Must
            take parameters `pm` (a (# of models,)-array of model probabilities),
            `py` (a (# of models, # of response values, # of importance samples, #
            of candidate designs)-array of response likelihoods), and `ptheta` (a (#
            of models, # of importance samples)-array of importance weights), and
            return a (# of candidate designs,)-array of global utility values.
        model1, model2, model3,... : array_like
            Allowed hypothesis classes. Each model must be an instance of
            `pybad.models.Model`.

        Returns
        -------
        utility_values : (# of candidate designs,)-array
            Global utility values for each candidate design.

        Examples
        --------
        Initialize allowed hypotheses as `Model` instances corresponding to the
        power-law and exponential models of memory retention:

        >>> from pybad.memory_retention import *
        >>> POW, EXP = init_models()

        Calculate the mutual information utility for parameter estimation for lag
        times between 0. and 100.:

        >>> U(np.arange(0., 101.)[:,None], u_parameterEstimation, POW, EXP)

        Calculate the mutual information utility for model selection:

        >>> U(np.arange(0., 101.)[:,None], u_modelSelection, POW, EXP)

        Calculate the mutual information utility for total entropy:

        >>> U(np.arange(0., 101.)[:,None], u_totalEntropy, POW, EXP)

    u_modelSelection(pm, py, ptheta)
        Model selection utility function described by [1]:

        u(x, y, θ, m) = log( p(m | y, x) / p(m) )

        References
        ----------
        .. [1] Cavagnaro, D. R., Myung, J. I., Pitt, M. A., and Kujala, J. V.
               (2010). Adaptive Design Optimization: A Mutual Information-Based
               Approach to Model Discrimination in Cognitive Science.
               *Neural Computation, 22*.

    u_parameterEstimation(pm, py, ptheta)
        Parameter estimation utility function:

        u(x, y, θ, m) = log( p(θ | y, x, m) / p(θ | m) )

        Following [1], the global utility is obtained by taking an average of the
        global utility *conditional on each candidate model*, weighted by the
        corresponding model probability.

        References
        ----------
        .. [1] Cavagnaro, D. R., Aranovich, G. J., McClure, S. M., Pitt, M. A., and
               Myung, J. I. (2016). On the functional form of temporal discounting:
               An optimized adaptive test. *Journal of Risk and Uncertainty, 52*.

    u_totalEntropy(pm, py, ptheta)
        Total entropy utility function introduced by [1]:

        u(x, y, θ, m) = log( p(θ, m | y, x) / p(θ, m) )

        References
        ----------
        .. [1] Borth, D. M. (1975). A Total Entropy Criterion for the Dual Problem
               of Model Discrimination and Parameter Estimation. *Journal of the
               Royal Statistical Society: Series B (Methodological), 37*(1).

    update_models(y, x, *models)
        Update the model probabilities and parameter distributions for a set of
        models, on the basis of a binary response to a single stimulus.

        Parameters
        ----------
        y : {int, (1,)-array, (1, 1)-array}
            Observed response (0 or 1).
        d : {int, (# of design attributes,)-array,
                (1, # of design attributes)-array}
            Design that produced response `y`.
        model1, model2, model3,... : array_like
            Allowed hypothesis classes. Each model must be an instance of
            `pybad.models.Model`.

        Returns
        -------
        None
            The `p_m` and `dist` attributes of each item in `models` are updated to
            reflect the calculated model probabilities and parameter distributions,
            respectively.

        Examples
        --------
        Initialize allowed hypotheses as `Model` instances corresponding to the
        power-law and exponential models of memory retention:

        >>> from pybad.memory_retention import *
        >>> POW, EXP = init_models()

        Update the model probabilities and corresponding parameter distributions on
        the basis of observing retention after a lag of 10 seconds:

        >>> update_models(1, 10., POW, EXP)
