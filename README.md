Implementation of Bayesian adaptive design, as described in Sloman, S. J. (2022). Chapter 2: Fast and Accurate Bayesian Adaptive Design via Sequential Importance Sampling. In *Towards Robust Bayesian Adaptive Design Methods for the Study of Human Behavior* [Unpublished dissertation draft].

# Installation

Necessary dependencies can be installed in a dedicated `conda` environment by running:

`conda env create --file pybad.yml`

The package itself can then be made available from anywhere in the environment by running:

`conda develop .`

Configurable settings can be adjusted in the file `settings`.

# Basic usage

Initialize a model corresponding to the power-law model of memory retention:

```
>>> from scipy.stats import beta
>>> from pybad.models import BinaryClassModel
>>> from pybad.tasks.memory_retention import pow_f
>>> POW = BinaryClassModel(
>>>     f=pow_f, param_bounds=[(0.,1.),(0.,1.)], 
>>>     prior=beta(a=[2.,1.], b=[1.,4.])
>>> )
```

Select the stimulus that will, in expectation, be most informative about the parameters of the model:

```
>>> import numpy as np
>>> from pybad.bad.sbinom import *
>>> candidate_designs = np.arange(101)[:,None]
>>> UU = U(candidate_designs, u_parameterEstimation, POW)
>>> optimal_design = candidate_designs[[UU.argmax()],:]
>>> optimal_design
array([[0]])
```

Simulate a response under a particular set of generating parameters and update the parameter distribution based on the response:

```
>>> y = pow_f(a=.9025, b=.4861, t=optimal_design)
>>> update_models(y, optimal_design, POW)
```

For examples of multi-trial experiments for the purposes of model selection, see the Jupyter notebooks in the folder `demos`. Parameter settings for this example taken from Cavagnaro, D. R., Myung, J. I., Pitt, M. A., and Kujala, J. V. (2010). Adaptive Design Optimization: A Mutual Information-Based Approach to Model Discrimination in Cognitive Science. *Neural Computation, 22*.