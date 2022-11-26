"""Core code to implement procedure described in Cavagnaro, D. R., Myung, J. I.,
Pitt, M. A., and Kujala, J. V. (2010). Adaptive Design Optimization: A Mutual
Information-Based Approach to Model Discrimination in Cognitive Science. *Neural 
Computation, 22*.
"""

from numba import jit
import numpy as np
from scipy.stats import bernoulli, beta
from pybad.bad.sbinom import *
from pybad.config import *
from pybad.experiments import *
from pybad.models import BinaryClassModel

drawer = f"{paths['drawer']}/memory_retention"

nopython = settings["nopython"]

###### Prediction functions ######

@jit(nopython=nopython)
def pow_f(a, b, t):
    return a * (t[:,0] + 1)**(-b)

@jit(nopython=nopython)
def exp_f(a, b, t):
    return a * np.exp(-b * t[:,0])

###### Response functions ######

def response_function(true_model=pow_f, true_params=(.9025,.4861)):
    return lambda x: bernoulli.rvs(true_model(*true_params, x))[:,None]

###### Initialize models ######

# Specifications from Cavagnaro et al. (2010)
def init_models():
    POW = BinaryClassModel(
        f=pow_f, param_bounds=[(0.,1.),(0.,1.)], 
        prior=beta(a=[2.,1.], b=[1.,4.]), p_m=.5
    )
    EXP = BinaryClassModel(
        f=exp_f, param_bounds=[(0.,1.),(0.,1.)], 
        prior=beta(a=[2.,1.], b=[1.,80.]), p_m=.5
    )
    return [ POW, EXP ]

###### Experimental designs ######

def ado(n, ntrials_left, *models):
    d = U(np.arange(101)[:,None], u_modelSelection, *models).argmax()
    return np.repeat(np.atleast_2d(d), 10, axis=0)

def random(n, ntrials_left, *models):
    return np.repeat(np.random.choice(np.arange(101)), 10)[:,None]

def fixed(n, ntrials_left, *models):
    return np.array([0, 1, 2, 4, 7, 12, 21, 35, 59, 99])[:,None]

###### Helper functions to run demo notebook ######

def update_models_from_multiple_obs(y, x, *models):
    for yy, xx in zip(y, x):
        update_models(yy, xx, *models)

def run_experiment(design_method, response_function, logname):
    logging.basicConfig(
        filename=f"{drawer}/{logname}.log", level=logging.INFO
    )
    return run(
        design_method=design_method, response_function=response_function,
        models=init_models(), ntrials=10, path=f"{drawer}/{logname}", 
        update=update_models_from_multiple_obs
    )