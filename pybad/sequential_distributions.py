from copy import deepcopy
import logging
import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal as mvn

def _check_scalar_in_bounds(x, bounds):
    return x >= bounds[0] and x <= bounds[1]
_check_array_in_bounds = np.vectorize(
    _check_scalar_in_bounds, signature="(),(n)->()"
)
_check_matrix_in_bounds = np.vectorize(
    _check_array_in_bounds, excluded=[1], signature="(n)->(n)"
)

class FlexiArray(object):
    def __init__(self):
        self.values = np.empty((0,0))
    
    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, value):
        self._values = value
    
    def __iadd__(self, x):
        if self._values.shape[0] == 0:
            self._values = x
        else:
            self._values = np.append(self._values, x, axis=0)
        return self
    
    def asarray(self):
        return self._values

class SequentialDistribution(object):
    """Generic class for building representations of the parameter distribution.
    """
    
    def __init__(
        self, likelihood, prior, init=True, nparams=None, nsamples=10000,
        param_bounds=None, weighted=True
    ):
        self._distribution = deepcopy(prior)
        self._is_prior = True
        self._likelihood = likelihood
        self._nsamples = nsamples
        self._prior = prior
        self._weighted = weighted
        self.nparams = nparams
        self.param_bounds = param_bounds
        self.X = FlexiArray()
        self.Y = FlexiArray()
        if init:
            self.samples, self.W = self.sample()
            
    @property
    def nparams(self):
        return self._nparams 
        
    @nparams.setter
    def nparams(self, value):
        self._nparams = value
        if isinstance(self._nparams, type(None)):
            _rvs = self._distribution.rvs()
            if hasattr(_rvs, "__iter__"):
                self._nparams = _rvs.shape[0]
            else:
                logging.info("""Setting `Sampler.nparams` = 1.
                Manually pass `nparams` as argument to override.""")
                self._nparams = 1
                
    @property
    def param_bounds(self):
        return self._param_bounds
    
    @param_bounds.setter
    def param_bounds(self, value):
        if isinstance(value, type(None)):
            self._param_bounds = np.vstack((
                np.repeat(-np.inf, self._nparams), 
                np.repeat(np.inf, self._nparams)
            )).T
        else:
            self._param_bounds = np.array(value)
    
    @property
    def samples(self):
        return self._samples
    
    @samples.setter
    def samples(self, value):
        self._samples = value
    
    @property
    def W(self):
        return self._W
    
    @W.setter
    def W(self, value):
        self._W = value
        
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, value):
        self._X = value
    
    @property
    def Y(self):
        return self._Y
    
    @Y.setter
    def Y(self, value):
        self._Y = value
    
    def _vectorized_likelihood(self, y, x, samples=None):
        if isinstance(samples, type(None)):
            samples = self._samples
        return np.apply_along_axis(self._likelihood, 1, samples, y, x)
        
    # Pseudo- (potentially unnormalized) pdf for the distribution
    def _ppdf(self, x):
        return self._distribution.pdf(x)
        
    def _prior_weights(self, y, x, zeroproof=False):
        ll = self._vectorized_likelihood(y, x)
        wghts = ll * self._W
        if wghts.sum() == 0. or zeroproof:
            wghts += np.finfo(float).eps
        return wghts / wghts.sum()
    
    def _sample(self):
        return self._distribution.rvs(size=(self._nsamples,self._nparams))
    
    def sample(self, throwaway=False):
        samples = self._sample()
        wghts = self._set_weights(
            samples, weighted=(~self._is_prior & self._weighted)
        )
        if not throwaway:
            self.samples = samples
            self.W = wghts
        return samples, wghts
        
    def sample_from_prior(self):
        return self._prior.rvs(size=(self._nsamples,self._nparams))
    
    def _set_equal_weights(self, samples):
        return np.ones_like(samples[:,0]) / samples.shape[0]
    
    # OK to ignore normalizing constants for P and Q
    # [ (P*a) / (Q*b) ] / \sum_{ (P*a) / (Q*b) } = ( P / Q ) / \sum_{ P / Q }
    # If the likelihood is 0., this will return a vector of `nan`s for the
    # importance weights. This may cause failure later, but should be up to the
    # user to decide how to deal with models under which all observations are
    # impossible.
    def _set_importance_weights(self, samples, Q=None):
        if isinstance(Q, type(None)):
            Q = self._ppdf(samples)
        P = self._prior.pdf(samples).prod(axis=1)
        P *= self._vectorized_likelihood(
            self._Y.asarray(), self._X.asarray(), samples=samples
        )
        wghts = P / Q
        wghts[Q == 0.] = 0.
        return wghts / wghts.sum()
    
    def _set_weights(self, samples, Q=None, weighted=True):
        if not weighted:
            return self._set_equal_weights(samples)
        return self._set_importance_weights(samples, Q=Q)
    
    def update(self, y, x):
        self._X += x
        self._Y += y
        self._distribution, self._samples, self._W = self._update(y, x)
        self._is_prior = False
    
class Grid(SequentialDistribution):
    """Implements a grid approximation.
    
    Parameters
    ----------
    likelihood : callable
        The likelihood function for the represented model. Must take parameters
        `theta` (an (`nparams`,)-array_like sequence of parameter values), `y`
        (a (# of designs, # of responses per design)-array of observed 
        responses), and `d` (a (# of designs, # of design attributes)-array of
        corresponding stimuli), and return the likelihood of `y` given `theta`
        and `d`. When `self` is initialized from a `pybad.models.Model` object,
        this is set to `pybad.models.Model().likelihood_fixed_param`.
    prior : `scipy.stats.rv_generic`
        Prior parameter distribution.
    init : bool, optional
        Indicates whether `self` should initialize with importance samples and 
        importance weights, default is True.
    nparams : int, optional
        The number of model parameters, default is None, in which case the
        number of parameters is inferred from the output of `prior.rvs()`.
    param_bounds : array_like, optional
        Iterable of length `nparams`, of the form [(lower_1,upper_1),
        (lower_2,upper_2),(lower_3,upper_3),...], default is [(-inf,inf),
        (-inf,inf),(-inf,inf)...].
    nsamples : int, optional
        The number of importance samples used to represent the parameter 
        distribution, default is `res**nparams`. If `res` is not specified,
        default is 10000.
    weighted : bool, optional
        Indicates whether to use importance sampling. If False, the weights of
        all importance samples are set to equal and never updated, effectively
        treating the samples as a direct representation of the parameter
        distribution, default is True.
    res : int, optional
        The resolution of the grid. Must be specified if `nsamples` is None. If
        `nsamples` is specified, default is `nsamples**(1. / nparams)`.
        
    Attributes
    ----------
    samples : (`nsamples`, `nparams`)-array
        Importance samples.
    W : (`nsamples`,)-array
        Importance weights.
    X : `FlexiArray`
        Iterable containing the history of x-values used to update the
        distribution.
    Y : `FlexiArray`
        Iterable containing the history of y-values used to update the
        distribution.
    
    Methods
    -------
    sample
    sample_from_prior
    update
    
    Examples
    --------
    Initialize a `Model` instance representing the exponential model of delay
    discounting, with the parameter space represented as a grid with a
    resolution of 4000:
    
    >>> from pybad.sequential_distributions import Grid
    >>> from pybad.intertemporal_choice import *
    >>> EXP = BinaryClassModel(
    >>>     f=exp_const, param_bounds=[(.0005,.2)], 
    >>>     prior=uniform(loc=[.0005], scale=[.1995]), dist=Grid, res=4000
    >>> )
    """
    
    def __init__(self, **kwargs):
        _init = kwargs.get("init", True)
        kwargs["init"] = False
        self._res = kwargs.get("res", None)
        if "res" in kwargs.keys():
            del kwargs["res"]
        super().__init__(**kwargs)
        if isinstance(self._nsamples, type(None)):
            assert isinstance(self._res, int)
            self._nsamples = self._res**self._nparams
        if isinstance(self._res, type(None)):
            assert isinstance(self._nsamples, int)
            _res = self._nsamples**(1./self._nparams)
            assert _res == int(_res)
            self._res = int(_res)
        assert self._nsamples == self._res**self._nparams
        self._quantiles = np.linspace(
            np.finfo(float).eps, 1-np.finfo(float).eps, self._res
        )
        qstack = self._stack_list([self._quantiles]*self._nparams)
        self._grid = self._distribution.ppf(qstack)
        if _init:
            self.samples, self.W = self.sample()
        
    def _ppdf(self, x):
        return self._distribution.pdf(x).prod(axis=1)
        
    def _sample(self):
        return self._grid
    
    def _set_weights(self, samples, weighted=None):
        return self._set_equal_weights(samples)
    
    def _stack_list(self, alist):
        grid = np.meshgrid(*np.stack(alist))
        return np.stack([ g.ravel() for g in grid ], axis=1)
    
    def _update(self, y, x):
        ll = self._vectorized_likelihood(y, x)
        if ll.sum() == 0.:
            return self._distribution, self._samples, self._W
        wghts = self._W * ll
        return self._distribution, self._samples, wghts / wghts.sum()
    
class Gaussian(SequentialDistribution):
    """Implements a Gaussian approximation to the parameter distribution as the
    importance distribution, as described by [1].
    
    Parameters
    ----------
    likelihood : callable
        The likelihood function for the represented model. Must take parameters
        `theta` (an (`nparams`,)-array_like sequence of parameter values), `y`
        (a (# of designs, # of responses per design)-array of observed 
        responses), and `d` (a (# of designs, # of design attributes)-array of
        corresponding stimuli), and return the likelihood of `y` given `theta`
        and `d`. When `self` is initialized from a `pybad.models.Model` object,
        this is set to `pybad.models.Model().likelihood_fixed_param`.
    prior : `scipy.stats.rv_generic`
        Prior parameter distribution.
    init : bool, optional
        Indicates whether `self` should initialize with importance samples and 
        importance weights, default is True.
    nparams : int, optional
        The number of model parameters, default is None, in which case the
        number of parameters is inferred from the output of `prior.rvs()`.
    param_bounds : array_like, optional
        Iterable of length `nparams`, of the form [(lower_1,upper_1),
        (lower_2,upper_2),(lower_3,upper_3),...], default is [(-inf,inf),
        (-inf,inf),(-inf,inf)...].
    nsamples : int, optional
        The number of importance samples used to represent the parameter 
        distribution, default is `res**nparams`. If `res` is not specified,
        default is 10000.
    weighted : bool, optional
        Indicates whether to use importance sampling. If False, the weights of
        all importance samples are set to equal and never updated, effectively
        treating the samples as a direct representation of the parameter
        distribution, default is True.
    scale_cov : scalar, optional
        Factor by which to scale the covariance matrix of the importance
        distribution, default is 1.
        
    Attributes
    ----------
    samples : (`nsamples`, `nparams`)-array
        Importance samples.
    W : (`nsamples`,)-array
        Importance weights.
    X : `FlexiArray`
        Iterable containing the history of x-values used to update the
        distribution.
    Y : `FlexiArray`
        Iterable containing the history of y-values used to update the
        distribution.
        
    Methods
    -------
    sample
    sample_from_prior
    update
    
    References
    ----------
    .. [1] E.G. Ryan, C.C. Drovandi, and A.N. Pettitt, Fully Bayesian 
           Experimental Designs for Pharmacokinetic Studies. Entropy 17 (2015).
    
    Examples
    --------
    Initialize a `Model` instance representing the power-law model of memory
    retention, with the parameter importance distribution a Gaussian
    approximation with the covariance inflated by a factor of 2.:
    
    >>> from pybad.sequential_distributions import Gaussian
    >>> from pybad.memory_retention import *
    >>> POW = BinaryClassModel(
    >>>     f=pow_f, param_bounds=[(0.,1.),(0.,1.)], 
    >>>     prior=beta(a=[2.,1.], b=[1.,4.]), dist=Gaussian, scale_cov=2.
    >>> )
    """
    
    def __init__(self, **kwargs):
        self._scale_cov = kwargs["scale_cov"]
        del kwargs["scale_cov"]
        super().__init__(**kwargs)
    
    def _gaussian_approximation(self, x, w, scale_cov=1.):
        mu = x.T@w
        sigma = np.cov(x.T, aweights=w)
        return mvn(mean=mu, cov=scale_cov*sigma, allow_singular=True)
    
    def _resample_mvn(self, mvndist):
        theta = mvndist.rvs(size=self._nsamples).reshape(
            (self._nsamples,self._nparams)
        )
        out_of_bounds = ~_check_matrix_in_bounds(theta, self._param_bounds)
        while np.any(out_of_bounds):
            logging.info("""{}% of samples are out of bounds. 
            Resampling...""".format(
                np.sum(out_of_bounds) / np.prod(out_of_bounds.shape) * 100
            ))
            ridx, cidx = np.where(out_of_bounds)
            theta[ridx,:] = mvndist.rvs(size=ridx.shape[0])
            out_of_bounds = ~_check_matrix_in_bounds(theta, self._param_bounds)
        return theta
    
    def _sample(self, mvndist=None):
        if self._is_prior:
            return self.sample_from_prior()
        if isintance(mvndist, type(None)):
            mvndist = self._distribution
        return self._resample_mvn(mvndist)
    
    def _update(self, y, x):
        w = self._prior_weights(y, x, zeroproof=True)
        post = self._gaussian_approximation(
            self._samples, w, scale_cov=self._scale_cov
        )
        samples = self._resample_mvn(post)
        return post, samples, self._set_weights(
            samples, Q=post.pdf(samples), weighted=self._weighted
        )
    
class KDE(SequentialDistribution):
    """Constructs a Gaussian kernel density approximation to the parameter 
    distribution as the importance distribution. Builds on the top of
    `scipy.stats.gaussian_kde` [1].
    
    Parameters
    ----------
    likelihood : callable
        The likelihood function for the represented model. Must take parameters
        `theta` (an (`nparams`,)-array_like sequence of parameter values), `y`
        (a (# of designs, # of responses per design)-array of observed 
        responses), and `d` (a (# of designs, # of design attributes)-array of
        corresponding stimuli), and return the likelihood of `y` given `theta`
        and `d`. When `self` is initialized from a `pybad.models.Model` object,
        this is set to `pybad.models.Model().likelihood_fixed_param`.
    prior : `scipy.stats.rv_generic`
        Prior parameter distribution.
    init : bool, optional
        Indicates whether `self` should initialize with importance samples and 
        importance weights, default is True.
    nparams : int, optional
        The number of model parameters, default is None, in which case the
        number of parameters is inferred from the output of `prior.rvs()`.
    param_bounds : array_like, optional
        Iterable of length `nparams`, of the form [(lower_1,upper_1),
        (lower_2,upper_2),(lower_3,upper_3),...], default is [(-inf,inf),
        (-inf,inf),(-inf,inf)...].
    nsamples : int, optional
        The number of importance samples used to represent the parameter 
        distribution, default is `res**nparams`. If `res` is not specified,
        default is 10000.
    weighted : bool, optional
        Indicates whether to use importance sampling. If False, the weights of
        all importance samples are set to equal and never updated, effectively
        treating the samples as a direct representation of the parameter
        distribution, default is True.
    bw_method : str, scalar or callable, optional
        Determines the bandwidth of the KDE. Argument to 
        `scipy.stats.gaussian_kde`, default is "scott", corresponding to Scott's
        factor [2].
        
    Attributes
    ----------
    samples : (`nsamples`, `nparams`)-array
        Importance samples.
    W : (`nsamples`,)-array
        Importance weights.
    X : `FlexiArray`
        Iterable containing the history of x-values used to update the
        distribution.
    Y : `FlexiArray`
        Iterable containing the history of y-values used to update the
        distribution.
        
    Methods
    -------
    sample
    sample_from_prior
    update
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    .. [2] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [3] B.W. Silverman, "Density Estimation for Statistics and Data 
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    
    Examples
    --------
    Initialize a `Model` instance representing the power-law model of memory
    retention, with the parameter importance distribution a kernel density
    estimate using Silverman's rule [3] to estimate the bandwidth:
    
    >>> from pybad.sequential_distributions import KDE
    >>> from pybad.memory_retention import *
    >>> POW = BinaryClassModel(
    >>>     f=pow_f, param_bounds=[(0.,1.),(0.,1.)], 
    >>>     prior=beta(a=[2.,1.], b=[1.,4.]), dist=KDE, bw_method="silverman"
    >>> )
    """
    
    def __init__(self, **kwargs):
        self._bw_method = kwargs.get("bw_method")
        if "bw_method" in kwargs.keys():
            del kwargs["bw_method"]
        super().__init__(**kwargs)
    
    def _ppdf(self, x):
        return self._distribution.pdf(x.T)
    
    def _resample_kde(self, kde=None):
        if isinstance(kde, type(None)):
            kde = self._distribution
        theta = kde.resample(size=self._nsamples).T
        out_of_bounds = ~_check_matrix_in_bounds(theta, self._param_bounds)
        while np.any(out_of_bounds):
            logging.info("""{}% of samples are out of bounds. 
            Resampling...""".format(
                np.sum(out_of_bounds) / np.prod(out_of_bounds.shape) * 100
            ))
            ridx, cidx = np.where(out_of_bounds)
            theta[ridx,:] = kde.resample(size=ridx.shape[0]).T
            out_of_bounds = ~_check_matrix_in_bounds(theta, self._param_bounds)
        return theta
    
    def _sample(self):
        if self._is_prior:
            return self.sample_from_prior()
        return self._resample_kde()
        
    def _update(self, y, x):
        w = self._prior_weights(y, x, zeroproof=True)
        post = gaussian_kde(
            self._samples.T, bw_method=self._bw_method, weights=w
        )
        samples = self._resample_kde(kde=post)
        return post, samples, self._set_weights(
            samples, Q=post.pdf(samples.T), weighted=self._weighted
        )
