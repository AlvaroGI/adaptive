# -*- coding: utf-8 -*-

import sys
from copy import deepcopy

import numpy as np

from adaptive.learner.average_mixin import DataPoint, add_average_mixin
from adaptive.learner.learner1D import Learner1D, _get_neighbors_from_list, loss_manager
from adaptive.notebook_integration import ensure_holoviews


class AverageLearner1D(Learner1D):
    """Learns and predicts a noisy function 'f:ℝ → ℝ^N'.

    New parameters (wrt Learner1D)
    ------------------------------
    delta : float
        The minimum value of the mean error.
    """
    def __init__(self, function, bounds, loss_per_interval=None, delta=0.1, min_samples=5):
        super().__init__(function, bounds, loss_per_interval)

        self._data_samples = {} # This SortedDict contains all samples f(x) for each x
                                # in the form {x0:[f_0(x0), f_1(x0), ...]}
        self._number_samples = {} # This SortedDict contains the number of samples
                                  # for each x in the form {n0: x0, n1: x1}
        self.min_samples = min_samples

        self._mean_error = {} # This SortedDict contains the estimation errors for
                              # each fbar(x) in the form {estimation_error(x0): x0}
        self.delta = delta

        print('__init__ ok!')


    def ask(self, n, tell_pending=True):
        """Return 'n' points that are expected to maximally reduce the loss."""
        assert isinstance(self._mean_error, dict)
        assert isinstance(self._number_samples, dict)

        if not self._mean_error.__len__():
            points, loss_improvements = self._ask_points_without_adding(n)
        elif self._number_samples.peekitem(0) < self.min_samples:
            _, x = self._mean_error.popitem(0)
            print('NEW ASK')
            points, loss_improvements = self._ask_for_more_samples(x,n)
        elif self._mean_error.peekitem(-1) > self.delta:
            _, x = self._mean_error.popitem(-1)
            print('NEW ASK')
            points, loss_improvements = self._ask_for_more_samples(x,n)
        else:
            points, loss_improvements = self._ask_points_without_adding(n)

        if tell_pending:
            for p in points:
                self.tell_pending(p)

        return points, loss_improvements

    def _ask_for_more_samples(self,x,n):
        points = [x] * n
        loss_improvements = [0] * n
        print('loss_improvements not implemented yet')
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
        if x in self._data:
            print('Not implemented yet!')
            return
        else:
            super().tell(x,y)

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
