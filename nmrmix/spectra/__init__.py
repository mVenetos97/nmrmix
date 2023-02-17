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

from nmrmix.nmr import parse_coupled_SimpleSpinSystem, couple_SimpleSpinSystem


def render_spect(spin_system, shifts, eval_points, render_width):
    """
    Shifts: locations of shifts (length N)
    intenisty: intensity of shifts (reflecting duplicate shift values, like in a methyl)
    eval_points: evaluation points
    render_width: how wide are the peaks? [fixme: make this more physical]


    return: length(eval_points) observation
    """
    sites_count = len(spin_system.sites)
    temp_spin_system = spin_system.copy(deep=True)

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
        -((updated_shifts[:, jnp.newaxis] - eval_points) ** 2) / render_width**2
    )

    y_sum = (y * updated_intenisities[:, jnp.newaxis]).sum(axis=0)

    # I felt as though I should normalize the output of the
    # spectrum as this will be multiplied by an intensity in model
    return y_sum / y_sum.max()


def model(
    pred_shifts_mu, pred_shifts_sigma, spin_system, eval_points, y, high=None, low=None
):

    MIXTURE_COMPONENTS, NUM_POSSIBLE_SHIFTS = pred_shifts_mu.shape
    # EVAL_N = eval_points.shape[0]
    assert len(y) == len(eval_points)

    mixture_weights = numpyro.sample(
        "mixture_weights", dist.HalfCauchy(scale=1 * jnp.ones(MIXTURE_COMPONENTS))
    )

    shifts = numpyro.sample(
        "shifts",
        dist.TruncatedDistribution(
            dist.Normal(pred_shifts_mu, pred_shifts_sigma), low=low, high=high
        ),
    )

    A = [
        render_spect(sim, shift, x, RENDER_WIDTH)
        for sim, shift in zip(spin_system, shifts)
    ]
    A = jnp.array(A)

    mu = A.T @ mixture_weights

    return numpyro.sample("obs", dist.Normal(mu, 0.1), obs=y)
