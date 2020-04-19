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


class AverageLearner1D(Learner1D):
    """Learns and predicts a noisy function 'f:ℝ → ℝ^N'.

    New parameters (wrt Learner1D)
    ------------------------------
    min_samples : int
        The minimum number of samples at each point x.
    strategy : int (1-3)
        Strategy chosen to sample new points or re-sample.
        1 = Check the error in the loss.
        2 = Check the error in the mean values.
        3 = Take 'moving average' (this strategy never re-samples).
    delta : float
        The minimum value of the mean error.
    alfa : float
        The size of the interval of confidence of the estimate of the mean
        is 1-2*alfa
    """
    def __init__(self, function, bounds, loss_per_interval=None, min_samples=3, strategy=None, delta=0.1, alfa=0.025):
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
        self.delta = delta

        self.alfa = alfa

        if not strategy:
                raise ValueError('Strategy not specified.')
        elif strategy>3:
                raise ValueError('Incorrect strategy (should be 1, 2 or 3)')
        else:
                self.strategy = strategy

        random.seed(2)

        #print('__init__ ok!')


    def ask(self, n, tell_pending=True):
        """Return 'n' points that are expected to maximally reduce the loss."""
        assert isinstance(self._error_in_mean, dict)
        assert isinstance(self._number_samples, dict)

        if self.strategy==3:
            raise ValueError('Strategy 3 not implemented.')
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
        # If there are at least 2 points and the loss difference is too large,
        # sample n/(2*(nth_neighbors+1) times each point of the interval # REVIEW
        elif self.strategy==1 and self._losses_diff.peekitem(0)[1] > self.delta*max(1,self._scale[1]):
            x1, x2 = self._losses_diff.peekitem(0)[0]
            # We invest half of the samples on each point. In case n is odd,
            # in order to not introduce any asymmetric bias, we choose at random
            # which x will be sampled more times
            if random.randint(0,1):
                    points1, loss_improvements1 = self._ask_for_more_samples(x1,n//2)
                    points2, loss_improvements2 = self._ask_for_more_samples(x2,n-n//2)
            else:
                    points1, loss_improvements1 = self._ask_for_more_samples(x1,n-n//2)
                    points2, loss_improvements2 = self._ask_for_more_samples(x2,n//2)
            points = points1+points2
            loss_improvements = loss_improvements1+loss_improvements2
        # If the error on the estimate of the mean is too large at x,
        # ask for more samples at x
        elif self.strategy==2 and self._error_in_mean.peekitem(0)[1] > self.delta:
                x = self._error_in_mean.peekitem(0)[0]
                points, loss_improvements = self._ask_for_more_samples(x,n)
        # Otherwise, sample new points
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
            self._data[x] = y
            self.pending_points.discard(x)
            super()._update_data_structures(x, y)
            self._data_samples[x] = [y]
            self._number_samples[x] = 1
            self._undersampled_points.add(x)
            self._error_in_mean[x] = np.inf # REVIEW: should be np.inf?
            self._update_losses_diff(x)
        # If re-sampled data point:
        else:
            self._update_data(x,y)
            self.pending_points.discard(x)
            self._update_data_structures_resampling(x, y)
            self._update_losses_diff(x)


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
            #print(n)
            self._undersampled_points.discard(x)

        # We compute the error in the estimation of the mean as
        # the std of the mean multiplied by a t-Student factor to ensure that
        # the mean value lies within the correct interval of confidence
        y_avg = self._data[x]
        variance_in_mean = sum( [(yj-y_avg)**2 for yj in self._data_samples[x]] )/(n-1)
        t_student = tstud.ppf(1.0 - self.alfa, df=n-1)
        self._error_in_mean[x] = t_student*(variance_in_mean/n)**0.5

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

    def _update_losses_diff(self, x, real=True):
        """Update all _losses_diff that depend on x.
           The current implementation only updates real points."""
        # When we add a new point x, we should update the losses
        # (x_left, x_right) are the "real" neighbors of 'x'.
        x_left, x_right = self._find_neighbors(x, self.neighbors)

        if real:
            # We need to update all interpolated losses in the interval
            # (x_left, x), (x, x_right) and the nth_neighbors nearest
            # neighboring intervals. Since the addition of the
            # point 'x' could change their loss.
            for ival in _get_intervals(x, self.neighbors, self.nth_neighbors):
                self._update_loss_diff_in_interval(*ival)

            # Since 'x' is in between (x_left, x_right),
            # we get rid of the interval.
            self._losses_diff.pop((x_left, x_right), None)
        else:
            print('_update_losses_diff with real=False not implemented...')

    def _update_loss_diff_in_interval(self, x_left, x_right):
        if x_left is None or x_right is None:
            return
        loss_diff = self._get_loss_diff_in_interval(x_left, x_right)
        self._losses_diff[x_left, x_right] = loss_diff

    def _get_loss_diff_in_interval(self, x_left, x_right):
        assert x_left is not None and x_right is not None

        if x_right - x_left < self._dx_eps:
            return 0

        nn = self.nth_neighbors
        i = self.neighbors.index(x_left)
        start = i - nn
        end = i + nn + 2

        xs = [self._get_point_by_index(i) for i in range(start, end)]
        # Note that we apply a random sign!!
        ys = [self.data.get(x, None)+random_sign()*self._error_in_mean.get(x, None) for x in xs]

        xs_scaled = tuple(self._scale_x(x) for x in xs)
        ys_scaled = tuple(self._scale_y(y) for y in ys)

        loss_IC = self.loss_per_interval(xs_scaled, ys_scaled) # This loss is calculated with
                                                               # extremes values of ys inside
                                                               # its confidence interval

        loss = self.losses[(xs[0],xs[1])] # Recall: the keys of this dict are tuples, not lists

        return abs(loss-loss_IC)/loss


    def tell_many(self,x,y):
        #print('Not implemented yet.')
        return

    def total_samples(self):
        '''Returns the total number of samples'''
        _, ns = zip(*self._number_samples.items())
        return sum(ns)

def random_sign():
    return 1 if random.random() < 0.5 else -1

def error_in_mean_initializer():
    def sorting_rule(key, value):
        return -value
    return sortedcollections.ItemSortedDict(sorting_rule, sortedcontainers.SortedDict())

    # def _ask_points_without_adding(self, n):
    #     points, loss_improvements = super()._ask_points_without_adding(n)
    #     points = [(p, 0) for p in points]
    #     return points, loss_improvements

    # def _get_neighbor_mapping_new_points(self, points):
    #     return {
    #         p: [n for n in self._find_neighbors(p, self.neighbors) if n is not None]
    #         for p in points
    #     }

    # def _get_neighbor_mapping_existing_points(self):
    #     return {k: [x for x in v if x is not None] for k, v in self.neighbors.items()}

    # def tell(self, x_seed, y):
    #     x, seed = x_seed
    #
    #     # either it is a float/int, if not, try casting to a np.array
    #     if not isinstance(y, (float, int)):
    #         y = np.asarray(y, dtype=float)
    #
    #     self._add_to_data(x_seed, y)
    #     self._remove_from_to_pending(x_seed)
    #     self._update_data_structures(x, y)

    # def tell_pending(self, x_seed):
    #     x, seed = x_seed
    #
    #     self._add_to_pending(x_seed)
    #
    #     if x not in self.neighbors_combined:
    #         # If 'x' already exists then there is not need to update.
    #         self._update_neighbors(x, self.neighbors_combined)
    #         self._update_losses(x, real=False)

    # def tell_many(self, xs, ys, *, force=False):
    #     if not force and not (len(xs) > 0.5 * len(self._data) and len(xs) > 2):
    #         # Only run this more efficient method if there are
    #         # at least 2 points and the amount of points added are
    #         # at least half of the number of points already in 'data'.
    #         # These "magic numbers" are somewhat arbitrary.
    #         for x, dp in zip(xs, ys):
    #             for seed, y in dp.items():
    #                 self.tell((x, seed), y)
    #         return
    #
    #     # Add data points
    #     self._data.update(zip(xs, ys))
    #     for x, dp in zip(xs, ys):
    #         if x in self.pending_points:
    #             seeds = dp.keys()
    #             self.pending_points[x].difference_update(seeds)
    #             if len(self.pending_points[x]) == 0:
    #                 # Remove if pending_points[x] is an empty set.
    #                 del self.pending_points[x]
    #
    #     # Below is the same as 'Learner1D.tell_many'.
    #
    #     # Get all data as numpy arrays
    #     points = np.array(list(self._data.keys()))
    #     values = np.array(list(self.data.values()))
    #     points_pending = np.array(list(self.pending_points))
    #     points_combined = np.hstack([points_pending, points])
    #
    #     # Generate neighbors
    #     self.neighbors = _get_neighbors_from_list(points)
    #     self.neighbors_combined = _get_neighbors_from_list(points_combined)
    #
    #     # Update scale
    #     self._bbox[0] = [points_combined.min(), points_combined.max()]
    #     self._bbox[1] = [values.min(axis=0), values.max(axis=0)]
    #     self._scale[0] = self._bbox[0][1] - self._bbox[0][0]
    #     self._scale[1] = np.max(self._bbox[1][1] - self._bbox[1][0])
    #     self._oldscale = deepcopy(self._scale)
    #
    #     # Find the intervals for which the losses should be calculated.
    #     intervals, intervals_combined = [
    #         [(x_m, x_r) for x_m, (x_l, x_r) in neighbors.items()][:-1]
    #         for neighbors in (self.neighbors, self.neighbors_combined)
    #     ]
    #
    #     # The the losses for the "real" intervals.
    #     self.losses = loss_manager(self._scale[0])
    #     for ival in intervals:
    #         self.losses[ival] = self._get_loss_in_interval(*ival)
    #
    #     # List with "real" intervals that have interpolated intervals inside
    #     to_interpolate = []
    #
    #     self.losses_combined = loss_manager(self._scale[0])
    #     for ival in intervals_combined:
    #         # If this interval exists in 'losses' then copy it otherwise
    #         # calculate it.
    #         if ival in reversed(self.losses):
    #             self.losses_combined[ival] = self.losses[ival]
    #         else:
    #             # Set all losses to inf now, later they might be udpdated if the
    #             # interval appears to be inside a real interval.
    #             self.losses_combined[ival] = np.inf
    #             x_left, x_right = ival
    #             a, b = to_interpolate[-1] if to_interpolate else (None, None)
    #             if b == x_left and (a, b) not in self.losses:
    #                 # join (a, b) and (x_left, x_right) → (a, x_right)
    #                 to_interpolate[-1] = (a, x_right)
    #             else:
    #                 to_interpolate.append((x_left, x_right))
    #
    #     for ival in to_interpolate:
    #         if ival in reversed(self.losses):
    #             # If this interval does not exist it should already
    #             # have an inf loss.
    #             self._update_interpolated_loss_in_interval(*ival)

    # def remove_unfinished(self):
    #     self.pending_points = {}
    #     self.losses_combined = deepcopy(self.losses)
    #     self.neighbors_combined = deepcopy(self.neighbors)

    # def plot(self, *, with_sem=True):
    #     hv = ensure_holoviews()
    #
    #     xs = sorted(self._data.keys())
    #     vdims = ["mean", "standard_error", "std", "n"]
    #     values = [[getattr(self._data[x], attr) for x in xs] for attr in vdims]
    #     scatter = hv.Scatter((xs, *values), vdims=vdims)
    #
    #     if not with_sem:
    #         plot = scatter.opts(plot=dict(tools=["hover"]))
    #     else:
    #         ys, sems, *_ = values
    #         err = [x if x < sys.float_info.max else np.nan for x in sems]
    #         spread = hv.Spread((xs, ys, err))
    #         plot = scatter * spread
    #     return plot.opts(hv.opts.Scatter(tools=["hover"]))

    # def _set_data(self, data):
    #     # change dict -> DataPoint, because the points are saved using dicts
    #     data = {k: DataPoint(v) for k, v in data.items()}
    #     self.tell_many(data.keys(), data.values())
