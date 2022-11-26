"""Core code to implement procedure described in Cavagnaro, D. R., 
Aranovich, G. J., McClure, S. M., Pitt, M. A., and Myung, J. I. (2016). On the
functional form of temporal discounting: An optimized adaptive test. *Journal of
Risk and Uncertainty, 52*.
"""

from numba import jit
import numpy as np
from scipy.stats import bernoulli, uniform
from pybad.bad.sbinom import *
from pybad.config import *
from pybad.experiments import *
from pybad.models import BinaryClassModel

drawer = f"{paths['drawer']}/intertemporal_choice"
ADO_GRID = np.loadtxt(f"{drawer}/cavagnaro2016")
KIRBY_GRID = np.loadtxt(f"{drawer}/kirby1999")

nopython = settings["nopython"]

###### Link functions ######

@jit(nopython=nopython)
def true_and_error(A_SS, D_SS, A_LL, D_LL, error_rate=.25):
    return .5 + (.5-error_rate)*(np.sign(A_SS*D_SS - A_LL*D_LL))

@jit(nopython=nopython)
def logistic(A_SS, D_SS, A_LL, D_LL, epsilon=2):
    return 1 / (1 + np.exp(epsilon*(A_LL*D_LL - A_SS*D_SS)))

###### Delay discounting functions ######

@jit(nopython=nopython)
def exponential(r, t):
    return np.exp(-r*t)

@jit(nopython=nopython)
def hyperbolic(k, t):
    return 1 / (1 + k*t)

@jit(nopython=nopython)
def generalized_hyperbolic(k, s, t):
    return 1 / (1 + k*t)**s

@jit(nopython=nopython)
def beta_delta(beta, delta, t):
    return (beta*delta**t)**(t != 0)

@jit(nopython=nopython)
def double_exponential(omega, r, s, t):
    return omega*np.exp(-r*t) + (1 - omega)*np.exp(-s*t)

@jit(nopython=nopython)
def constant_sensitivity(r, s, t):
    return np.exp(-(r*t)**s)

###### Prediction functions ######

def _base_pfunc(discount_function, link_function, x, *params):
    D_SS = discount_function(*params, x[:,1])
    D_LL = discount_function(*params, x[:,3])
    return link_function(x[:,0], D_SS, x[:,2], D_LL)

# All `pybad.models.Model`s need at least one free parameter, although in this
# case it has no effect on the probability function.
def coin_flip(theta, x):
    return .5*np.ones_like(x[:,0])

### Constant error specification ###

def exp_const(r, x):
    return _base_pfunc(exponential, true_and_error, x, r)

def hyp_const(k, x):
    return _base_pfunc(hyperbolic, true_and_error, x, k)

def gm_const(k, s, x):
    return _base_pfunc(generalized_hyperbolic, true_and_error, x, k, s)

def bd_const(beta, delta, x):
    return _base_pfunc(beta_delta, true_and_error, x, beta, delta)

def de_const(omega, r, s, x):
    return _base_pfunc(double_exponential, true_and_error, x, omega, r, s)
    
def cs_const(r, s, x):
    return _base_pfunc(constant_sensitivity, true_and_error, x, r, s)

### Logistic error specification ###

def exp_logistic(r, x):
    return _base_pfunc(exponential, logistic, x, r)

def hyp_logistic(k, x):
    return _base_pfunc(hyperbolic, logistic, x, k)

def gm_logistic(k, s, x):
    return _base_pfunc(generalized_hyperbolic, logistic, x, k, s)

def bd_logistic(beta, delta, x):
    return _base_pfunc(beta_delta, logistic, x, beta, delta)

def de_logistic(omega, r, s, x):
    return _base_pfunc(double_exponential, logistic, x, omega, r, s)
    
def cs_logistic(r, s, x):
    return _base_pfunc(constant_sensitivity, logistic, x, r, s)

###### Response function ######

def response_function(true_model=cs_logistic, true_params=(.025,.4)):
    return lambda x: bernoulli.rvs(true_model(*true_params, x))

###### Initialize models ######

# Specifications from Cavagnaro et al. (2016)
def init_models():
    Exp = BinaryClassModel(
        f=exp_const, param_bounds=[.0005,.2], 
        prior=uniform(loc=[.0005], scale=[.1995]), p_m=1./7.
    )
    Hyp = BinaryClassModel(
        f=hyp_const, param_bounds=[.001,.1], 
        prior=uniform(loc=[.001], scale=[.099]), p_m=1./7.,
    )
    CS = BinaryClassModel(
        f=cs_const, param_bounds=[[.0005,.1], [.15,1.5]], 
        prior=uniform(loc=[.0005,.15], scale=[.1495,1.35]), p_m=1./7.
    )
    GM = BinaryClassModel(
        f=gm_const, param_bounds=[[.0001,1.], [.1,2.]], 
        prior=uniform(loc=[.0001,.1], scale=[.999,1.9]), p_m=1./7.
    )
    BD = BinaryClassModel(
        f=bd_const, param_bounds=[[0.,1.], [0.,1.]], 
        prior=uniform(loc=[0.,0.], scale=[1.,1.]), p_m=1./7.
    )
    DE = BinaryClassModel(
        f=de_const, param_bounds=[[0.,.8], [.8,1.], [0.,1.]],
        prior=uniform(loc=[0.,.8,0.], scale=[.8,.2,1.]), p_m=1./7.
    )
    # All `pybad.models.Model`s need at least one free parameter, although in 
    # this case it has no effect on the probability function.
    CF = BinaryClassModel(
        f=coin_flip, prior=uniform(loc=[0.], scale=[1.]), p_m=1./7.
    )
    
    return [ Exp, Hyp, CS, GM, BD, DE, CF ]

###### Experimental designs ######

def ado(n, ntrials_left, *models):
    UU = U(ADO_GRID, u_modelSelection, *models)
    return ADO_GRID[None,np.argmax(UU),:]
    
def geometric(n, ntrials_left, *models):
    return ADO_GRID[None,np.random.randint(ADO_GRID.shape[0]),:]

def kirby(n, ntrials_left, *models):
    return KIRBY_GRID[None,n%27,:]
    
def random(n, ntrials_left, *models):
    xS = np.random.uniform(1, 20)
    tS = np.random.uniform(0, 40)
    xL = np.random.uniform(xS+1, 30)
    tL = np.random.uniform(tS+1, 80)
    return np.array([[ xS, tS, xL, tL ]])

###### Helper function to run demo notebook ######

def run_experiment(design_method, response_function, logname):
    logging.basicConfig(
        filename=f"{drawer}/{logname}.log", level=logging.INFO
    )
    return run(
        design_method=design_method, response_function=response_function, 
        models=init_models(), ntrials=81, path=f"{drawer}/{logname}"
    )