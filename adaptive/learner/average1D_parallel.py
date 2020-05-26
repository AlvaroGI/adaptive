# -*- coding: utf-8 -*-

import sys
import itertools
import math
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import sortedcollections
import sortedcontainers
from scipy.stats import t as tstud
import random

from adaptive.learner.average_mixin import DataPoint, add_average_mixin
from adaptive.learner.learner1D import Learner1D, _get_neighbors_from_list, loss_manager, _get_intervals
from adaptive.notebook_integration import ensure_holoviews


class AverageLearner1D_parallel(Learner1D):
    """Learns and predicts a noisy function 'f:ℝ → ℝ^N'.

    New parameters (wrt Learner1D)
    ------------------------------
    min_samples : int (>1)
        Minimum number of samples at each point x. Each new point is initially
        sampled min_samples times.
    neighbor_sampling: float (>0, <1)
        Each new point is initially sampled min_samples times.
    delta : float
        The minimum value of the mean error.
    alfa : float
        The size of the interval of confidence of the estimate of the mean
        is 1-2*alfa
    max_samples: int
    min_Delta_g
    """
    def __init__(self, function, bounds, loss_per_interval=None, min_samples=3, neighbor_sampling=0.3, delta=0.1, alfa=0.005, max_samples=np.inf, min_Delta_g=0):
        super().__init__(function, bounds, loss_per_interval)

        self._data_samples = sortedcontainers.SortedDict() # This SortedDict contains all samples f(x) for each x
                                                           # in the form {x0:[f_0(x0), f_1(x0), ...]}
        self._number_samples = sortedcontainers.SortedDict() # This SortedDict contains the number of samples
                                                             # for each x in the form {x0: n0, x1: n1, etc.}
        self._undersampled_points = set() # This set contains the points x that
                                          # are undersampled
        self.min_samples = min_samples

        # Relative error between the real losses and the losses calculated
        # using extreme values of the intervals of confidence
        self._losses_diff = loss_manager(self._scale[0]) # {x: loss_diff(x)}

        self._error_in_mean = error_in_mean_initializer() # This SortedDict contains the estimation errors for
                                                          # each fbar(x) in the form {x0: estimation_error(x0)}


        self._rescaled_error_in_mean = error_in_mean_initializer() # {xi: Delta_gi/min(Delta_g_{i},Delta_g_{i-1})}
        self._interval_sizes = error_in_mean_initializer() # {xi: xii-xi}
        self._min_Delta_g = min_Delta_g
        self._max_samples = max_samples

        self.delta = delta

        self.alfa = alfa

        self.neighbor_sampling = neighbor_sampling


    def ask(self, n, tell_pending=True):
        """Return 'n' points that are expected to maximally reduce the loss."""
        assert isinstance(self._error_in_mean, dict)
        assert isinstance(self._number_samples, dict)

        # If self.data contains no points, proceed as in Learner1D
        if not self.data.__len__():
            points, loss_improvements = self._ask_points_without_adding(n)
        # If some point is under-sampled, invest the next samples on it
        elif len(self._undersampled_points):
            x = self._undersampled_points.pop()
            self._undersampled_points.add(x)
            points, loss_improvements = self._ask_for_more_samples(x,n)
        # If only 1 point was sampled, sample a new one
        elif self.data.__len__() == 1:
            points, loss_improvements = self._ask_points_without_adding(n)
        # Else
        else:
            x, resc_error = self._rescaled_error_in_mean.peekitem(0)
            # Resampling condition
            if (resc_error > self.delta):
                points, loss_improvements = self._ask_for_more_samples(x,n)
            # Sample new point
            else:
                points, loss_improvements = self._ask_points_without_adding(n)


        if tell_pending:
            #print('tell_pending must be redesigned carefully')
            for p in points:
                self.tell_pending(p)

        return points, loss_improvements

    def _ask_for_more_samples(self,x,n):
        points = [x] * n
        loss_improvements = [0] * n
        #print('loss_improvements not implemented yet')
        return points, loss_improvements

    def tell_pending(self, x):
        # Even if x is in self._data, we can sample it again
        # if x in self._data:
        #     # The point is already evaluated before
        #     return
        self.pending_points.add(x) # Note that a set cannot contain duplicates
        if x not in self._data:
            self._update_neighbors(x, self.neighbors_combined)
            self._update_losses(x, real=False)

    def tell(self, x, y):
        if y is None:
            raise TypeError(
                "Y-value may not be None, use learner.tell_pending(x)"
                "to indicate that this value is currently being calculated"
            )

        # either it is a float/int, if not, try casting to a np.array
        if not isinstance(y, (float, int)):
            y = np.asarray(y, dtype=float)

        '''The next if/else can be alternatively included in a different
           definition of _update_data'''
        # If new data point, operate as in Learner1D
        if not self._data.__contains__(x):
            self._data_samples[x] = [y]
            self._data[x] = y
            self.pending_points.discard(x)
            super()._update_data_structures(x, y)
            self._number_samples[x] = 1
            self._undersampled_points.add(x)
            self._error_in_mean[x] = np.inf # REVIEW: should be np.inf?
            self._rescaled_error_in_mean[x] = np.inf # REVIEW: should be np.inf?
            self._update_interval_sizes(x)
            self._update_rescaled_error_in_mean(x,'new')
        # If re-sampled data point:
        else:
            self._update_data(x,y)
            self.pending_points.discard(x)
            self._update_data_structures_resampling(x, y)

    def _update_rescaled_error_in_mean(self,x,point_type):
        '''Updates self._rescaled_error_in_mean; point_type must be "new" or
           "resampled". '''
        assert point_type=='new' or point_type=='resampled', 'point_type must be "new" or "resampled"'

        xleft, xright = self.neighbors[x]
        if xleft is None and xright is None:
            return

        if (xleft is None):
            dleft = self._interval_sizes[x]
        else:
            dleft = self._interval_sizes[xleft]
            if self._rescaled_error_in_mean.__contains__(xleft):
                xll = self.neighbors[xleft][0]
                if xll is None:
                    self._rescaled_error_in_mean[xleft] = self._error_in_mean[xleft] / self._interval_sizes[xleft]
                else:
                    self._rescaled_error_in_mean[xleft] = self._error_in_mean[xleft] / min(self._interval_sizes[xll],
                                                                                           self._interval_sizes[xleft])
        if (xright is None):
            dright = self._interval_sizes[xleft]
        else:
            dright = self._interval_sizes[x]
            if self._rescaled_error_in_mean.__contains__(xright):
                xrr = self.neighbors[xright][1]
                if xrr is None:
                    self._rescaled_error_in_mean[xright] = self._error_in_mean[xright] / self._interval_sizes[x]
                else:
                    self._rescaled_error_in_mean[xright] = self._error_in_mean[xright] / min(self._interval_sizes[x],
                                                                                             self._interval_sizes[xright])

        if point_type=='resampled':
            self._rescaled_error_in_mean[x] = self._error_in_mean[x] / min(dleft,dright)
        return

    def _update_interval_sizes(self,x):
        neighbors = self.neighbors[x]
        if neighbors[0] is not None:
        #    self._interval_sizes[neighbors[0]] = x-neighbors[0]
            self._interval_sizes[neighbors[0]] = ((x-neighbors[0])**2 + (self.data[x]-self.data[neighbors[0]])**2)**0.5
        if neighbors[1] is not None:
            self._interval_sizes[x] = ((neighbors[1]-x)**2 + (self.data[neighbors[1]]-self.data[x])**2)**0.5
        return

    def _update_data(self,x,y):
        '''This function is only used if self._data contains x'''
        n = len(self._data_samples[x])
        new_average = self._data[x]*n/(n+1) + y/(n+1)
        self._data[x] = new_average

    def _update_data_structures_resampling(self, x, y):
        '''This function is only used if self._data already contains x'''
        # No need to check that x is inside the bounds
        # No need to update neighbors

        # We have to update _data_samples, _number_samples,
        # _undersampled_points
        self._data_samples[x].append(y)

        self._number_samples[x] = self._number_samples[x]+1
        n = self._number_samples[x]

        if (x in self._undersampled_points) and (n >= self.min_samples):
            xleft, xright = self.neighbors[x]
            n = self._number_samples[x]

            if xleft and xright:
                nneighbor = 0.5*(self._number_samples[xleft] + self._number_samples[xright])
            elif xleft:
                nneighbor = self._number_samples[xleft]
            elif xright:
                nneighbor = self._number_samples[xright]
            else:
                nneighbor = 0
            if n > self.neighbor_sampling * nneighbor:
                self._undersampled_points.discard(x)

        # We compute the error in the estimation of the mean as
        # the std of the mean multiplied by a t-Student factor to ensure that
        # the mean value lies within the correct interval of confidence
        y_avg = self._data[x]
        variance_in_mean = sum( [(yj-y_avg)**2 for yj in self._data_samples[x]] )/(n-1)
        t_student = tstud.ppf(1.0 - self.alfa, df=n-1)
        self._error_in_mean[x] = t_student*(variance_in_mean/n)**0.5
        self._update_interval_sizes(x)
        self._update_rescaled_error_in_mean(x,'resampled')

        if (self._rescaled_error_in_mean.__contains__(x)
            and (self._error_in_mean[x] <= self._min_Delta_g or self._number_samples[x] >= self._max_samples)):
            _ = self._rescaled_error_in_mean.pop(x)

        # We also need to update scale and losses
        super()._update_scale(x, y)
        self._update_losses_resampling(x, real=True) # REVIEW

        '''Is the following necessary?'''
        # If the scale has increased enough, recompute all losses.
        if self._scale[1] > self._recompute_losses_factor * self._oldscale[1]:
            for interval in reversed(self.losses):
                self._update_interpolated_loss_in_interval(*interval)

            self._oldscale = deepcopy(self._scale)

    def _update_losses_resampling(self, x, real=True):
        """Update all losses that depend on x, whenever the new point is a
           re-sampled point"""
        # (x_left, x_right) are the "real" neighbors of 'x'.
        x_left, x_right = self._find_neighbors(x, self.neighbors)
        # (a, b) are the neighbors of the combined interpolated
        # and "real" intervals.
        a, b = self._find_neighbors(x, self.neighbors_combined)

        if real:
            for ival in _get_intervals(x, self.neighbors, self.nth_neighbors):
                self._update_interpolated_loss_in_interval(*ival)
        elif x_left is not None and x_right is not None:
            # 'x' happens to be in between two real points,
            # so we can interpolate the losses.
            dx = x_right - x_left
            loss = self.losses[x_left, x_right]
            self.losses_combined[a, x] = (x - a) * loss / dx
            self.losses_combined[x, b] = (b - x) * loss / dx

        # (no real point left of x) or (no real point right of a)
        left_loss_is_unknown = (x_left is None) or (not real and x_right is None)
        if (a is not None) and left_loss_is_unknown:
            self.losses_combined[a, x] = float("inf")

        # (no real point right of x) or (no real point left of b)
        right_loss_is_unknown = (x_right is None) or (not real and x_left is None)
        if (b is not None) and right_loss_is_unknown:
            self.losses_combined[x, b] = float("inf")

    def tell_many(self,x,y):
        #print('Not implemented yet.')
        return

    def total_samples(self):
        '''Returns the total number of samples'''
        if not len(self._data):
            return 0
        else:
            _, ns = zip(*self._number_samples.items())
            return sum(ns)

def error_in_mean_initializer():
    '''This initialization orders the dictionary from large to small value'''
    def sorting_rule(key, value):
        return -value
    return sortedcollections.ItemSortedDict(sorting_rule, sortedcontainers.SortedDict())
