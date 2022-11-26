Help on module sequential_distributions:

NAME
    sequential_distributions

CLASSES
    builtins.object
        FlexiArray
        SequentialDistribution
            Gaussian
            Grid
            KDE

    class FlexiArray(builtins.object)
     |  Methods defined here:
     |
     |  __iadd__(self, x)
     |
     |  __init__(self)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  asarray(self)
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
     |  values

    class Gaussian(SequentialDistribution)
     |  Gaussian(**kwargs)
     |
     |  Implements a Gaussian approximation to the parameter distribution as the
     |  importance distribution, as described by [1].
     |
     |  Parameters
     |  ----------
     |  likelihood : callable
     |      The likelihood function for the represented model. Must take parameters
     |      `theta` (an (`nparams`,)-array_like sequence of parameter values), `y`
     |      (a (# of designs, # of responses per design)-array of observed
     |      responses), and `d` (a (# of designs, # of design attributes)-array of
     |      corresponding stimuli), and return the likelihood of `y` given `theta`
     |      and `d`. When `self` is initialized from a `pybad.models.Model` object,
     |      this is set to `pybad.models.Model().likelihood_fixed_param`.
     |  prior : `scipy.stats.rv_generic`
     |      Prior parameter distribution.
     |  init : bool, optional
     |      Indicates whether `self` should initialize with importance samples and
     |      importance weights, default is True.
     |  nparams : int, optional
     |      The number of model parameters, default is None, in which case the
     |      number of parameters is inferred from the output of `prior.rvs()`.
     |  param_bounds : array_like, optional
     |      Iterable of length `nparams`, of the form [(lower_1,upper_1),
     |      (lower_2,upper_2),(lower_3,upper_3),...], default is [(-inf,inf),
     |      (-inf,inf),(-inf,inf)...].
     |  nsamples : int, optional
     |      The number of importance samples used to represent the parameter
     |      distribution, default is `res**nparams`. If `res` is not specified,
     |      default is 10000.
     |  weighted : bool, optional
     |      Indicates whether to use importance sampling. If False, the weights of
     |      all importance samples are set to equal and never updated, effectively
     |      treating the samples as a direct representation of the parameter
     |      distribution, default is True.
     |  scale_cov : scalar, optional
     |      Factor by which to scale the covariance matrix of the importance
     |      distribution, default is 1.
     |
     |  Attributes
     |  ----------
     |  samples : (`nsamples`, `nparams`)-array
     |      Importance samples.
     |  W : (`nsamples`,)-array
     |      Importance weights.
     |  X : `FlexiArray`
     |      Iterable containing the history of x-values used to update the
     |      distribution.
     |  Y : `FlexiArray`
     |      Iterable containing the history of y-values used to update the
     |      distribution.
     |
     |  Methods
     |  -------
     |  sample
     |  sample_from_prior
     |  update
     |
     |  References
     |  ----------
     |  .. [1] E.G. Ryan, C.C. Drovandi, and A.N. Pettitt, Fully Bayesian
     |         Experimental Designs for Pharmacokinetic Studies. Entropy 17 (2015).
     |
     |  Examples
     |  --------
     |  Initialize a `Model` instance representing the power-law model of memory
     |  retention, with the parameter importance distribution a Gaussian
     |  approximation with the covariance inflated by a factor of 2.:
     |
     |  >>> from pybad.sequential_distributions import Gaussian
     |  >>> from pybad.memory_retention import *
     |  >>> POW = BinaryClassModel(
     |  >>>     f=pow_f, param_bounds=[(0.,1.),(0.,1.)],
     |  >>>     prior=beta(a=[2.,1.], b=[1.,4.]), dist=Gaussian, scale_cov=2.
     |  >>> )
     |
     |  Method resolution order:
     |      Gaussian
     |      SequentialDistribution
     |      builtins.object
     |
     |  Methods defined here:
     |
     |  __init__(self, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  ----------------------------------------------------------------------
     |  Methods inherited from SequentialDistribution:
     |
     |  sample(self, throwaway=False)
     |
     |  sample_from_prior(self)
     |
     |  update(self, y, x)
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from SequentialDistribution:
     |
     |  W
     |
     |  X
     |
     |  Y
     |
     |  __dict__
     |      dictionary for instance variables (if defined)
     |
     |  __weakref__
     |      list of weak references to the object (if defined)
     |
     |  nparams
     |
     |  param_bounds
     |
     |  samples

    class Grid(SequentialDistribution)
     |  Grid(**kwargs)
     |
     |  Implements a grid approximation.
     |
     |  Parameters
     |  ----------
     |  likelihood : callable
     |      The likelihood function for the represented model. Must take parameters
     |      `theta` (an (`nparams`,)-array_like sequence of parameter values), `y`
     |      (a (# of designs, # of responses per design)-array of observed
     |      responses), and `d` (a (# of designs, # of design attributes)-array of
     |      corresponding stimuli), and return the likelihood of `y` given `theta`
     |      and `d`. When `self` is initialized from a `pybad.models.Model` object,
     |      this is set to `pybad.models.Model().likelihood_fixed_param`.
     |  prior : `scipy.stats.rv_generic`
     |      Prior parameter distribution.
     |  init : bool, optional
     |      Indicates whether `self` should initialize with importance samples and
     |      importance weights, default is True.
     |  nparams : int, optional
     |      The number of model parameters, default is None, in which case the
     |      number of parameters is inferred from the output of `prior.rvs()`.
     |  param_bounds : array_like, optional
     |      Iterable of length `nparams`, of the form [(lower_1,upper_1),
     |      (lower_2,upper_2),(lower_3,upper_3),...], default is [(-inf,inf),
     |      (-inf,inf),(-inf,inf)...].
     |  nsamples : int, optional
     |      The number of importance samples used to represent the parameter
     |      distribution, default is `res**nparams`. If `res` is not specified,
     |      default is 10000.
     |  weighted : bool, optional
     |      Indicates whether to use importance sampling. If False, the weights of
     |      all importance samples are set to equal and never updated, effectively
     |      treating the samples as a direct representation of the parameter
     |      distribution, default is True.
     |  res : int, optional
     |      The resolution of the grid. Must be specified if `nsamples` is None. If
     |      `nsamples` is specified, default is `nsamples**(1. / nparams)`.
     |
     |  Attributes
     |  ----------
     |  samples : (`nsamples`, `nparams`)-array
     |      Importance samples.
     |  W : (`nsamples`,)-array
     |      Importance weights.
     |  X : `FlexiArray`
     |      Iterable containing the history of x-values used to update the
     |      distribution.
     |  Y : `FlexiArray`
     |      Iterable containing the history of y-values used to update the
     |      distribution.
     |
     |  Methods
     |  -------
     |  sample
     |  sample_from_prior
     |  update
     |
     |  Examples
     |  --------
     |  Initialize a `Model` instance representing the exponential model of delay
     |  discounting, with the parameter space represented as a grid with a
     |  resolution of 4000:
     |
     |  >>> from pybad.sequential_distributions import Grid
     |  >>> from pybad.intertemporal_choice import *
     |  >>> EXP = BinaryClassModel(
     |  >>>     f=exp_const, param_bounds=[(.0005,.2)],
     |  >>>     prior=uniform(loc=[.0005], scale=[.1995]), dist=Grid, res=4000
     |  >>> )
     |
     |  Method resolution order:
     |      Grid
     |      SequentialDistribution
     |      builtins.object
     |
     |  Methods defined here:
     |
     |  __init__(self, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  ----------------------------------------------------------------------
     |  Methods inherited from SequentialDistribution:
     |
     |  sample(self, throwaway=False)
     |
     |  sample_from_prior(self)
     |
     |  update(self, y, x)
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from SequentialDistribution:
     |
     |  W
     |
     |  X
     |
     |  Y
     |
     |  __dict__
     |      dictionary for instance variables (if defined)
     |
     |  __weakref__
     |      list of weak references to the object (if defined)
     |
     |  nparams
     |
     |  param_bounds
     |
     |  samples

    class KDE(SequentialDistribution)
     |  KDE(**kwargs)
     |
     |  Constructs a Gaussian kernel density approximation to the parameter
     |  distribution as the importance distribution. Builds on the top of
     |  `scipy.stats.gaussian_kde` [1].
     |
     |  Parameters
     |  ----------
     |  likelihood : callable
     |      The likelihood function for the represented model. Must take parameters
     |      `theta` (an (`nparams`,)-array_like sequence of parameter values), `y`
     |      (a (# of designs, # of responses per design)-array of observed
     |      responses), and `d` (a (# of designs, # of design attributes)-array of
     |      corresponding stimuli), and return the likelihood of `y` given `theta`
     |      and `d`. When `self` is initialized from a `pybad.models.Model` object,
     |      this is set to `pybad.models.Model().likelihood_fixed_param`.
     |  prior : `scipy.stats.rv_generic`
     |      Prior parameter distribution.
     |  init : bool, optional
     |      Indicates whether `self` should initialize with importance samples and
     |      importance weights, default is True.
     |  nparams : int, optional
     |      The number of model parameters, default is None, in which case the
     |      number of parameters is inferred from the output of `prior.rvs()`.
     |  param_bounds : array_like, optional
     |      Iterable of length `nparams`, of the form [(lower_1,upper_1),
     |      (lower_2,upper_2),(lower_3,upper_3),...], default is [(-inf,inf),
     |      (-inf,inf),(-inf,inf)...].
     |  nsamples : int, optional
     |      The number of importance samples used to represent the parameter
     |      distribution, default is `res**nparams`. If `res` is not specified,
     |      default is 10000.
     |  weighted : bool, optional
     |      Indicates whether to use importance sampling. If False, the weights of
     |      all importance samples are set to equal and never updated, effectively
     |      treating the samples as a direct representation of the parameter
     |      distribution, default is True.
     |  bw_method : str, scalar or callable, optional
     |      Determines the bandwidth of the KDE. Argument to
     |      `scipy.stats.gaussian_kde`, default is "scott", corresponding to Scott's
     |      factor [2].
     |
     |  Attributes
     |  ----------
     |  samples : (`nsamples`, `nparams`)-array
     |      Importance samples.
     |  W : (`nsamples`,)-array
     |      Importance weights.
     |  X : `FlexiArray`
     |      Iterable containing the history of x-values used to update the
     |      distribution.
     |  Y : `FlexiArray`
     |      Iterable containing the history of y-values used to update the
     |      distribution.
     |
     |  Methods
     |  -------
     |  sample
     |  sample_from_prior
     |  update
     |
     |  References
     |  ----------
     |  .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
     |  .. [2] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
     |         Visualization", John Wiley & Sons, New York, Chicester, 1992.
     |  .. [3] B.W. Silverman, "Density Estimation for Statistics and Data
     |         Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
     |         Chapman and Hall, London, 1986.
     |
     |  Examples
     |  --------
     |  Initialize a `Model` instance representing the power-law model of memory
     |  retention, with the parameter importance distribution a kernel density
     |  estimate using Silverman's rule [3] to estimate the bandwidth:
     |
     |  >>> from pybad.sequential_distributions import KDE
     |  >>> from pybad.memory_retention import *
     |  >>> POW = BinaryClassModel(
     |  >>>     f=pow_f, param_bounds=[(0.,1.),(0.,1.)],
     |  >>>     prior=beta(a=[2.,1.], b=[1.,4.]), dist=KDE, bw_method="silverman"
     |  >>> )
     |
     |  Method resolution order:
     |      KDE
     |      SequentialDistribution
     |      builtins.object
     |
     |  Methods defined here:
     |
     |  __init__(self, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  ----------------------------------------------------------------------
     |  Methods inherited from SequentialDistribution:
     |
     |  sample(self, throwaway=False)
     |
     |  sample_from_prior(self)
     |
     |  update(self, y, x)
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from SequentialDistribution:
     |
     |  W
     |
     |  X
     |
     |  Y
     |
     |  __dict__
     |      dictionary for instance variables (if defined)
     |
     |  __weakref__
     |      list of weak references to the object (if defined)
     |
     |  nparams
     |
     |  param_bounds
     |
     |  samples

    class SequentialDistribution(builtins.object)
     |  SequentialDistribution(likelihood, prior, init=True, nparams=None, nsamples=10000, param_bounds=None, weighted=True)
     |
     |  Generic class for building representations of the parameter distribution.
     |
     |  Methods defined here:
     |
     |  __init__(self, likelihood, prior, init=True, nparams=None, nsamples=10000, param_bounds=None, weighted=True)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  sample(self, throwaway=False)
     |
     |  sample_from_prior(self)
     |
     |  update(self, y, x)
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  W
     |
     |  X
     |
     |  Y
     |
     |  __dict__
     |      dictionary for instance variables (if defined)
     |
     |  __weakref__
     |      list of weak references to the object (if defined)
     |
     |  nparams
     |
     |  param_bounds
     |
     |  samples
