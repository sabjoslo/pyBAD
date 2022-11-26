Help on module models:

NAME
    models

CLASSES
    builtins.object
        Model
            BinaryClassModel

    class BinaryClassModel(Model)
     |  BinaryClassModel(**kwargs)
     |
     |  Represent a model whose probability function takes the form
     |  y ~ Bernoulli(f(x)), where f is a user-supplied parameter.
     |
     |  Parameters
     |  ----------
     |  f : callable
     |      The functional form associated with the model. Must take arguments
     |      `param1`, `param2`, `param3`,..., `design`.
     |  prior : `scipy.stats.rv_generic`
     |      Prior parameter distribution.
     |  nparams : int, optional
     |      The number of model parameters, default is None, in which case the
     |      number of parameters is inferred from the output of `prior.rvs()`.
     |  param_bounds : array_like, optional
     |      Iterable of length `nparams`, of the form [(lower_1,upper_1),
     |      (lower_2,upper_2),(lower_3,upper_3),...], default is [(-inf,inf),
     |      (-inf,inf),(-inf,inf)...].
     |  p_m : float, optional
     |      Initial model probability, default is 1.
     |  dist : callable, optional
     |      Represents and calculates expectations under the model's parameter
     |      distribution. Must be a class that inherits from
     |      `pybad.sequential_distributions.SequentialDistribution`, default is
     |      `KDE`, which will represent the posterior importance distribution using
     |      a weighted kernel density estimate.
     |  init : bool, optional
     |      Indicates whether `dist` should initialize itself with importance
     |      samples and importance weights, default is True.
     |  nsamples : int, optional
     |      The number of importance samples used to represent the parameter
     |      distribution, default is 10000.
     |  weighted : bool, optional
     |      Indicates whether to use importance sampling. If False, the weights of
     |      all importance samples are set to equal and never updated, effectively
     |      treating the samples as a direct representation of the parameter
     |      distribution, default is True.
     |
     |  Attributes
     |  ----------
     |  dist : `pybad.sequential_distributions.SequentialDistribution`
     |      Object representing the parameter distribution conditional on the
     |      corresponding model.
     |  p_m : float
     |      Probability of the model given all observations so far.
     |
     |  Methods
     |  -------
     |  likelihood
     |  likelihood_fixed_param
     |  log_likelihood
     |  predict
     |  predictive_dist
     |
     |  Examples
     |  --------
     |  Initialize a `Model` instance representing the power-law model of memory
     |  retention:
     |
     |  >>> from pybad.memory_retention import *
     |  >>> POW = BinaryClassModel(
     |  >>>     f=pow_f, param_bounds=[(0.,1.),(0.,1.)],
     |  >>>     prior=beta(a=[2.,1.], b=[1.,4.])
     |  >>> )
     |
     |  Calculate the likelihood of retention at a lag of 12.
     |
     |  >>> x = np.atleast_2d(12.)
     |  >>> y = np.atleast_2d(1)
     |  >>> POW.predict(x)
     |  >>> POW.likelihood(y)
     |  0.4309780798230697
     |
     |  Calculate the likelihood of retention if the parameters are a = .9025 and
     |  b = .4861.
     |
     |  >>> POW.likelihood_fixed_param([.9025, .4861], y, x)
     |  0.259393654028514
     |
     |  Calculate the likelihood of retention at lags 12, 13 and 100.
     |
     |  >>> POW.predictive_dist(
     |  >>>    np.array([12.,13.,100.])[:,None], np.ones((3,1)), posterior=False
     |  >>> )
     |  array([0.43097808, 0.42637121, 0.32992385])
     |
     |  Calculate how an observation of retention at a lag of 12 would affect the
     |  predicted likelihood of retention at 12, 13 and 100.
     |
     |  >>> POW.predictive_dist(np.array([12.,13.,100.])[:,None], np.ones((3,1)))
     |  array([0.54022798, 0.53580304, 0.43876012])
     |
     |  Method resolution order:
     |      BinaryClassModel
     |      Model
     |      builtins.object
     |
     |  Methods defined here:
     |
     |  __init__(self, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  likelihood(self, y)
     |
     |  likelihood_fixed_param(self, theta, y, x)
     |
     |  log_likelihood(self, y)
     |
     |  predictive_dist(self, x, y, posterior=True)
     |
     |  ----------------------------------------------------------------------
     |  Methods inherited from Model:
     |
     |  predict(self, x, ny)
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from Model:
     |
     |  __dict__
     |      dictionary for instance variables (if defined)
     |
     |  __weakref__
     |      list of weak references to the object (if defined)
     |
     |  p_m

    class Model(builtins.object)
     |  Model(f, prior, nparams=None, param_bounds=None, p_m=1.0, dist=<class 'pybad.sequential_distributions.KDE'>, **kwargs)
     |
     |  Generic class for building an object that implements a given probability
     |  function, and represents and can update a corresponding parameter
     |  distribution.
     |
     |  Methods defined here:
     |
     |  __init__(self, f, prior, nparams=None, param_bounds=None, p_m=1.0, dist=<class 'pybad.sequential_distributions.KDE'>, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  predict(self, x, ny)
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables (if defined)
     |
     |  __weakref__
     |      list of weak references to the object (if defined)
     |
     |  p_m
