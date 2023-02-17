"""
NMR module for Python
====================================================

This nmr module provides a set of useful functions for
the creation and simulation of molecules. The
implementation follows the design principles from the
mrsimulator python package with a few simplifications
for organic molecules.
"""
# version has to be specified at the start.
__author__ = "Maxwell C. Venetos"
__email__ = "mvenetos@berkeley.edu"
__license__ = "BSD License"
__maintainer__ = "Maxwell C. Venetos"
__status__ = "Beta"
__version__ = "0.1"

import numpy as np
import jax.numpy as jnp
import jax

from pydantic import BaseModel
from pydantic import validator
from typing import List
from typing import Union
from math import factorial


class SimpleCoupling(BaseModel):
    """
    site_index: list of the indicies of the site objects which are coupled
    isotropic_j: j-coupling frequency in Hz
    """

    site_index: List[int]
    isotropic_j: float = 0.0

    @validator("site_index", always=True)
    def validate_site_index(cls, v, *, values, **kwargs):
        if len(v) != 2:
            raise ValueError("Site index must a list of two integers.")
        if v[0] == v[1]:
            raise ValueError("The two site indexes must be unique integers.")
        return v


class SimpleSite(BaseModel):
    """
    isotope: specific isotope for NMR active atom
    isotropic_chemical_shift: value in ppm for chemical shift.
    Converted to a list when coupled
    multiplicity: Number of atoms responsible for this signal.
    Irrep in spin system
    intensity: the relative signal intensity. Taken to be multiplicity
    but updated upon coupling
    couple_flag: flag to determine if site has been coupled before.
    Present due to issues interfacing mrsimulator with JAX
    """

    isotope: str = "1H"
    isotropic_chemical_shift: Union[float, List] = 0.0
    multiplicity: int = 1
    intensity: int = 1
    couple_flag: bool = False
    symmetry_multiplier: int = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.intensity = self.multiplicity
        if isinstance(self.isotropic_chemical_shift, float):
            self.isotropic_chemical_shift = jnp.array([self.isotropic_chemical_shift])
        if isinstance(self.intensity, int):
            self.intensity = [self.intensity]

    def set_flag(self, state):
        self.couple_flag = state

    def set_intensity(self, intensity):
        self.intensity = intensity

    def set_isotropic_chemical_shift(self, isotropic_chemical_shift):
        self.isotropic_chemical_shift = isotropic_chemical_shift


class SimpleSpinSystem(BaseModel):
    """
    collection of sites and their couplings
    """

    sites: List[SimpleSite] = []
    couplings: List[SimpleCoupling] = []


#
# probably should be class memberfunctions
#


def comb(n, k):
    return factorial(n) / factorial(k) / factorial(n - k)


def comb_all(n):
    """
    creates the combinatorial intensities expected for a coupled site with n neighbors
    """
    combs = [comb(n, i) for i in range(n + 1)]
    return combs / np.sum(combs)


def ppm_to_Hz(ppm_val, carrier_frequency=500):
    """
    Converts a j-coupling frequency in ppm back to MHz
    carrier_frequency in Hz
    """
    return ppm_val * (carrier_frequency * 1000000) / 1e6


def Hz_to_ppm(Hz_val, carrier_frequency=500):
    """
    Converts a j-coupling frequency in Hz back to ppm
    carrier_frequency in MHz
    """
    return (Hz_val * 1e6) / (carrier_frequency * 1000000)


def couple_SimpleSpinSystem(SimpleSpinSystem, carrier_frequency=500):
    """
    Function to update isotropic_chemical_shift values in a site object
    to the list of shifts expected in a coupled spin system
    """
    if len(SimpleSpinSystem.couplings) == 0:
        return
    else:
        for coupling in SimpleSpinSystem.couplings:
            coupled_pair = coupling.site_index
            isotropic_j_ppm = Hz_to_ppm(coupling.isotropic_j, carrier_frequency)

            for i in coupled_pair:
                j = [s for s in coupled_pair if s != i][0]

                delta = SimpleSpinSystem.sites[i].isotropic_chemical_shift
                base_intensity = SimpleSpinSystem.sites[i].intensity

                neighbor_count = (
                    SimpleSpinSystem.sites[j].multiplicity
                    * SimpleSpinSystem.sites[j].symmetry_multiplier
                )
                centered_list = [
                    i - (neighbor_count - 1) / 2 for i in range(neighbor_count + 1)
                ]

                if SimpleSpinSystem.sites[i].couple_flag is False:
                    updated_freqs = [delta + n * isotropic_j_ppm for n in centered_list]
                    SimpleSpinSystem.sites[i].set_flag(state=True)
                else:
                    updated_freqs = [
                        d + n * isotropic_j_ppm for d in delta for n in centered_list
                    ]
                updated_intensities = [
                    base * i
                    for base in base_intensity
                    for i in comb_all(neighbor_count)
                ]

                SimpleSpinSystem.sites[i].set_isotropic_chemical_shift(updated_freqs)
                SimpleSpinSystem.sites[i].set_intensity(updated_intensities)


def parse_coupled_SimpleSpinSystem(SimpleSpinSystem):
    """
    Systematically parses a spin system to obtain the frequencies of each signal
    and the associated signal instensity
    """
    shifts = jnp.array([])
    couplings = []
    for site in SimpleSpinSystem.sites:
        shifts = jax.numpy.append(shifts, jnp.array(site.isotropic_chemical_shift))
        couplings += site.intensity

    return jnp.array(shifts), jnp.array(couplings)
