# -*- coding: utf-8 -*-
'''Auxiliary file used by error-averagelearner.ipynb'''

import random
from collections import defaultdict

import numpy as np

from adaptive.learner import AverageLearner1D
from adaptive.runner import simple as rsimple
from scipy.interpolate import interp1d
from scipy.integrate import quad

from random import random

def error_averagelearner1D_vs_uniform(fun,bounds,strategy,delta,min_samples,alfa,N,interp='linear'):
    '''Returns:
        -the L1-error between AverageLearner1D and the function without noise
        -the number of samples of the AverageLearner1D
        -the L1-error between a uniform learner and the function without noise
        -the number of samples of the uniform learner

        NOTE: the learners DO NOT run using async IO.
        '''
    # First, we check that the function has an input 'sigma' corresponding to
    # the standard deviation of the noise
    try:
        fun(0.1234,sigma=0)
    except:
        print('fun has no argument `sigma´ that accounts for the noise.')

    # Define and run AverageLearner1D
    learner = AverageLearner1D(fun, bounds=bounds,
                               strategy=strategy, delta=delta,
                               min_samples=min_samples, alfa=alfa)
    for _ in np.arange(N):
            xs, _ = learner.ask(1)
            for x in xs:
                y = learner.function(x)
                learner.tell(x, y)
    x_adaptive, y_adaptive = zip(*learner.data.items())
    N_adaptive = learner.total_samples()

    # Define and run uniform learner
    unif_samples_per_point = learner.total_samples()/len(learner.data)
    x_uniform = np.linspace(bounds[0],bounds[1],np.ceil(N_adaptive/unif_samples_per_point))
    learner_uniform = AverageLearner1D(fun, bounds=bounds,
                                       strategy=strategy, delta=delta,
                                       min_samples=min_samples, alfa=alfa)
    for _ in np.arange(unif_samples_per_point):
        for xx in x_uniform:
            yy = learner_uniform.function(xx)
            learner_uniform.tell(xx,yy)
    x_uniform, y_uniform = zip(*learner_uniform.data.items())
    N_uniform = learner_uniform.total_samples()

    # Create interpolators
    f_adaptive = interp1d(x_adaptive, y_adaptive, kind=interp, fill_value='extrapolate')
    f_uniform = interp1d(x_uniform, y_uniform, kind=interp, fill_value='extrapolate')

    def integrand_adaptive(x):
        return abs(f_adaptive(x)-fun(x,sigma=0))
    def integrand_uniform(x):
        return abs(f_uniform(x)-fun(x,sigma=0))

    e_adaptive, ee_adaptive = quad(integrand_adaptive,bounds[0],bounds[1])
    e_uniform, ee_uniform = quad(integrand_uniform,bounds[0],bounds[1])

    return e_adaptive, ee_adaptive, N_adaptive, e_uniform, ee_uniform, N_uniform
