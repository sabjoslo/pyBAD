from scipy.special import comb
from scipy.stats import norm
from pybad.sequential_distributions import *

class Model(object):
    """Generic class for building an object that implements a given probability
    function, and represents and can update a corresponding parameter 
    distribution.
    """
    
    def __init__(
        self, f, prior, nparams=None, param_bounds=None, p_m=1., dist=KDE, 
        **kwargs
    ):
        self._f = f
        self.p_m = p_m
        self.dist = dist(
            likelihood=self.likelihood_fixed_param, prior=prior, 
            nparams=nparams, param_bounds=param_bounds, **kwargs
        )
        signature = "{}->(n)".format(",".join(["()"] * self.dist.nparams))
        self._vectorized_over_params = np.vectorize(
            self._f, signature=signature, excluded=[self.dist.nparams]
        )
    
    @property
    def p_m(self):
        return self._p_m
    
    @p_m.setter
    def p_m(self, value):
        self._p_m = value
    
    def predict(self, x, ny):
        pred = self._vectorized_over_params(*self.dist.samples.T, x)
        self.pred = np.repeat(pred[:,:,None], ny, axis=-1)
        
class BinaryClassModel(Model):
    """Represent a model whose probability function takes the form 
    y ~ Bernoulli(f(x)), where f is a user-supplied parameter.
    
    Parameters
    ----------
    f : callable
        The functional form associated with the model. Must take arguments
        `param1`, `param2`, `param3`,..., `design`.
    prior : `scipy.stats.rv_generic`
        Prior parameter distribution.
    nparams : int, optional
        The number of model parameters, default is None, in which case the
        number of parameters is inferred from the output of `prior.rvs()`.
    param_bounds : array_like, optional
        Iterable of length `nparams`, of the form [(lower_1,upper_1),
        (lower_2,upper_2),(lower_3,upper_3),...], default is [(-inf,inf),
        (-inf,inf),(-inf,inf)...].
    p_m : float, optional
        Initial model probability, default is 1.
    dist : callable, optional
        Represents and calculates expectations under the model's parameter 
        distribution. Must be a class that inherits from
        `pybad.sequential_distributions.SequentialDistribution`, default is
        `KDE`, which will represent the posterior importance distribution using
        a weighted kernel density estimate.
    init : bool, optional
        Indicates whether `dist` should initialize itself with importance
        samples and importance weights, default is True.
    nsamples : int, optional
        The number of importance samples used to represent the parameter 
        distribution, default is 10000.
    weighted : bool, optional
        Indicates whether to use importance sampling. If False, the weights of
        all importance samples are set to equal and never updated, effectively
        treating the samples as a direct representation of the parameter
        distribution, default is True.
        
    Attributes
    ----------
    dist : `pybad.sequential_distributions.SequentialDistribution`
        Object representing the parameter distribution conditional on the
        corresponding model.
    p_m : float
        Probability of the model given all observations so far.
    
    Methods
    -------
    likelihood
    likelihood_fixed_param
    log_likelihood
    predict
    predictive_dist
    
    Examples
    --------
    Initialize a `Model` instance representing the power-law model of memory
    retention:
    
    >>> from pybad.memory_retention import *
    >>> POW = BinaryClassModel(
    >>>     f=pow_f, param_bounds=[(0.,1.),(0.,1.)], 
    >>>     prior=beta(a=[2.,1.], b=[1.,4.])
    >>> )
    
    Calculate the likelihood of retention at a lag of 12.
    
    >>> x = np.atleast_2d(12.)
    >>> y = np.atleast_2d(1)
    >>> POW.predict(x)
    >>> POW.likelihood(y)
    0.4309780798230697
    
    Calculate the likelihood of retention if the parameters are a = .9025 and
    b = .4861.
    
    >>> POW.likelihood_fixed_param([.9025, .4861], y, x)
    0.259393654028514
    
    Calculate the likelihood of retention at lags 12, 13 and 100.
    
    >>> POW.predictive_dist(
    >>>    np.array([12.,13.,100.])[:,None], np.ones((3,1)), posterior=False
    >>> )
    array([0.43097808, 0.42637121, 0.32992385])
    
    Calculate how an observation of retention at a lag of 12 would affect the
    predicted likelihood of retention at 12, 13 and 100.
    
    >>> POW.predictive_dist(np.array([12.,13.,100.])[:,None], np.ones((3,1)))
    array([0.54022798, 0.53580304, 0.43876012])
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _likelihood_unreduced(self, y):
        ll = self.pred.copy()
        ll[:,y==0] = 1-ll[:,y==0]
        return ll
    
    def likelihood(self, y):
        ll = self._likelihood_unreduced(y)
        return np.prod(ll, axis=(1,2))@self.dist.W
        
    def log_likelihood(self, y):
        ll = self._likelihood_unreduced(y)
        return np.log(ll + np.finfo(float).eps).sum(axis=(1,2))@self.dist.W
    
    # For use when calculating u_modelComparison and y is an
    # (n_samples x n_stims x n_y) matrix
    def _matrix_likelihood(self, y):
        ll = np.repeat(self.pred[:,None,:,:], y.shape[0], axis=1)
        ll[:,y==0] = 1-ll[:,y==0]
        return np.prod(ll, axis=(2,3))
    
    # For use when calculating u_parameterEstimation and y is an
    # (n_samples x n_stims x n_y) matrix
    def _vector_likelihood(self, y, normalized=True):
        ll = self.pred.copy()
        ll[y==0] = 1-ll[y==0]
        ll = np.prod(ll, axis=(1,2))
        if normalized:
            py = self._matrix_likelihood(y)
            ll /= py.T@self.dist.W
        return ll
    
    def likelihood_fixed_param(self, theta, y, x):
        ll = self._f(*theta, x)[:,None]
        ll = np.repeat(ll, y.shape[-1], axis=-1)
        ll[y==0] = 1-ll[y==0]
        return np.prod(ll)
    
    def predictive_dist(self, x, y, posterior=True):
        pred = self._vectorized_over_params(*self.dist.samples.T, x)
        if posterior:
            post = self._matrix_likelihood(y)
            post /= post.T@self.dist.W
            pred *= post
        return pred.T@self.dist.W