"""
NMR spectra module for Python
====================================================

This nmr module provides a set of useful functions for
the simulation of NMR spectra using the nmrmix.nmr
spinsystem objects.
"""
# version has to be specified at the start.
__author__ = "Maxwell C. Venetos"
__email__ = "mvenetos@berkeley.edu"
__license__ = "BSD License"
__maintainer__ = "Maxwell C. Venetos"
__status__ = "Beta"
__version__ = "0.1"

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from typing import List

from nmrmix.nmr import (
    parse_coupled_SimpleSpinSystem,
    couple_SimpleSpinSystem,
    SimpleSpinSystem,
)


class Model:
    def __init__(
        self,
        spin_systems: List[SimpleSpinSystem],
        pred_shifts_mu: List[List],
        pred_shifts_sigma: List[List],
        y: List[float],
        high: float = None,
        low: float = None,
        render_width: float = 0.015,
    ):
        self.spin_systems = spin_systems
        self.pred_shifts_mu = pred_shifts_mu
        self.pred_shifts_sigma = pred_shifts_sigma
        self.y = y
        self.high = high
        self.low = low
        self.render_width = render_width

    def render_spect(self, shifts, eval_points):
        """
        Shifts: locations of shifts (length N)
        eval_points: evaluation points, aka the ppm axis


        return: length(eval_points) observation
        """
        sites_count = len(self.spin_systems.sites)
        temp_spin_system = self.spin_systems.copy(deep=True)

        # insures that the list of shifts provided correspond to the correct couplings
        # additionally if too many shifts are provided we only take as many as the
        # spin system can handle (seen later when we use square matricies but have
        # different unmbers of sites per system)
        for i in range(sites_count):
            temp_spin_system.sites[i].isotropic_chemical_shift = shifts[i]
        couple_SimpleSpinSystem(temp_spin_system)
        updated_shifts, updated_intenisities = parse_coupled_SimpleSpinSystem(
            temp_spin_system
        )

        y = jnp.exp(
            -((updated_shifts[:, jnp.newaxis] - eval_points) ** 2)
            / self.render_width**2
        )

        y_sum = (y * updated_intenisities[:, jnp.newaxis]).sum(axis=0)

        # I felt as though I should normalize the output of the
        # spectrum as this will be multiplied by an intensity in model
        return y_sum / y_sum.max()

    def model(self, eval_points):

        mixtrue_components = self.pred_shifts_mu.shape
        # EVAL_N = eval_points.shape[0]
        assert len(self.y) == len(eval_points)

        mixture_weights = numpyro.sample(
            "mixture_weights", dist.HalfCauchy(scale=1 * jnp.ones(mixtrue_components))
        )

        shifts = numpyro.sample(
            "shifts",
            dist.TruncatedDistribution(
                dist.Normal(self.pred_shifts_mu, self.pred_shifts_sigma),
                low=self.low,
                high=self.high,
            ),
        )

        A = [
            self.render_spect(shift, eval_points)
            for sim, shift in zip(self.spin_systems, shifts)
        ]
        A = jnp.array(A)

        mu = A.T @ mixture_weights

        return numpyro.sample("obs", dist.Normal(mu, 0.1), obs=self.y)
