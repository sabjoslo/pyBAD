"""Implements utility functions and distribution update functions for sequential
binomial trials, i.e., for binary class models for which a single response 
y ~ Bernoulli(f(x)) is collected on each trial. 
"""

import numpy as np
from pybad.config import *

atol = settings["atol"]
rtol = settings["rtol"]

def u_modelSelection(pm, py, ptheta):
    """Model selection utility function described by [1]:
    
    u(x, y, \u03B8, m) = log( p(m | y, x) / p(m) )
    
    References
    ----------
    .. [1] Cavagnaro, D. R., Myung, J. I., Pitt, M. A., and Kujala, J. V. 
           (2010). Adaptive Design Optimization: A Mutual Information-Based 
           Approach to Model Discrimination in Cognitive Science. 
           *Neural Computation, 22*.
    """
    
    py = (py * ptheta[:,None,:,None]).sum(axis=2)
    uu = np.zeros((pm.shape[0],py.shape[-1]))
    for ii, p in enumerate(pm):
        u_ii = np.log(_update_p_m(pm, py, ii) / p)
        uu[ii] += np.where(py[ii,:] > 0, py[ii,:] * u_ii, 0).sum(axis=0)
    return pm[pm > 0.] @ uu[pm > 0.]

def u_parameterEstimation(pm, py, ptheta):
    """Parameter estimation utility function:
    
    u(x, y, \u03B8, m) = log( p(\u03B8 | y, x, m) / p(\u03B8 | m) )
    
    Following [1], the global utility is obtained by taking an average of the
    global utility *conditional on each candidate model*, weighted by the
    corresponding model probability.
    
    References
    ----------
    .. [1] Cavagnaro, D. R., Aranovich, G. J., McClure, S. M., Pitt, M. A., and
           Myung, J. I. (2016). On the functional form of temporal discounting:
           An optimized adaptive test. *Journal of Risk and Uncertainty, 52*.
    """
    
    uu = np.zeros((pm.shape[0],py.shape[-1]))
    for ii, p in enumerate(pm):
        mpy = (py[ii,:,:,:] * ptheta[ii,None,:,None]).sum(axis=1)[:,None,:]
        u_ii = np.log(py[ii,:,:,:] / mpy)
        u_ii = np.where(py[ii,:,:,:] > 0, py[ii,:,:,:] * u_ii, 0).sum(axis=0)
        uu[ii] += u_ii.T@ptheta[ii,:]
    return pm[pm > 0.] @ uu[pm > 0.]

def u_totalEntropy(pm, py, ptheta):
    """Total entropy utility function introduced by [1]:
    
    u(x, y, \u03B8, m) = log( p(\u03B8, m | y, x) / p(\u03B8, m) )
    
    References
    ----------
    .. [1] Borth, D. M. (1975). A Total Entropy Criterion for the Dual Problem 
           of Model Discrimination and Parameter Estimation. *Journal of the 
           Royal Statistical Society: Series B (Methodological), 37*(1).
    """
    
    margpy = (py * ptheta[:,None,:,None]).sum(axis=2)
    margpy = (pm[:,None,None] * margpy).sum(axis=0)
    uu = np.zeros((pm.shape[0],py.shape[-1]))
    for ii, p in enumerate(pm):
        u_ii = np.log(py[ii,:,:,:] / margpy[:,None,:])
        u_ii = np.where(py[ii,:,:,:] > 0, py[ii,:,:,:] * u_ii, 0).sum(axis=0)
        uu[ii] += u_ii.T@ptheta[ii,:]
    return pm[pm > 0.] @ uu[pm > 0.]

def U(designs, u, *models):
    """Calculate global utility for a set of candidate designs.
    
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
    """
    
    weights = np.stack([ m.dist.W for m in models ])
    pm = np.array([ m.p_m for m in models ])    
    py = np.zeros((
        len(models),2,models[0].dist.samples.shape[0],designs.shape[0]
    ))
    for mi, m in enumerate(models):
        p1 = m._vectorized_over_params(*m.dist.samples.T, designs)[None,:,:]
        py[mi,:,:,:] += np.vstack((1-p1, p1))
    return u(pm, py, ptheta=weights)

def _update_p_m(pm, py, ii, nanproof=True):
    if pm[ii] == 0.:
        return pm[ii]
    jj = np.delete(np.arange(pm.shape[0]), ii)
    bf = np.where(
        (py[jj,:,:] != 0) | (py[ii,:,:] != 0), py[jj,:,:] / py[ii,:,:], 1
    )
    # If the likelihood corresponding to a particular model is `nan`, that
    # should indicate that all observed response patterns were impossible
    # under that model.
    if nanproof:
        bf[np.isnan(bf)] = 0.
    # This should prevent `nan`s when multiplying `inf` by 0.
    bf[pm[jj] == 0.,:,:] = 0.
    return pm[ii] / (pm[ii] + (pm[jj,None,None] * bf).sum(axis=0))

update_p_m = np.vectorize(_update_p_m, excluded=[0,1])

def update_models(y, x, *models):
    """Update the model probabilities and parameter distributions for a set of
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
    """
    
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    for m in models:
        m.predict(x, 1)
    ll = [ m.likelihood(y*np.ones((1,1))) for m in models ]
    pm = np.array([ m.p_m for m in models ])
    py = np.array(ll)[:,None,None]
    post_p_m = update_p_m(pm, py, np.arange(len(models)))
    for ii, m in enumerate(models):
        m.dist.update(y, x)
        # If this condition is met, it means all the importance samples from the
        # updated distribution assign a probability of 0. to the history of 
        # observations. If a model assigns the history of observations a 
        # likelihood of 0., the corresponding posterior model probability should 
        # also be 0. In rare cases, this condition will be met and the 
        # corresponding posterior model probability is not exactly 0., 
        # indicating the history was possible in a region of the parameter space
        # sufficiently rare to have not been resampled. Here, I enforce the 
        # condition that the posterior model probability is 0., which avoids 
        # inconsistencies later. For sufficiently large numbers of importance 
        # samples, these cases should lead to inconsistencies on the order of 
        # floating point errors.
        if np.all(np.isnan(m.dist.W)):
            post_p_m[ii] = 0.
    # Don't renormalize so it can be detected if this leads to egregious
    # inconsistencies
    #post_p_m /= post_p_m.sum()
    assert np.isclose(post_p_m.sum(), 1., rtol=rtol, atol=atol)
    for ii, m in enumerate(models):
        m.p_m = post_p_m[ii]
