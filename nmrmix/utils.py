import numpy as np
from rdkit import Chem


def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def create_interval_list(
    shifts_pred_mu,
    ppm_span=1,
    pad=None,
):
    intervals = []
    for molecule in shifts_pred_mu:
        temp = []
        for shift in molecule:
            print("shift", shift)
            if shift != pad:
                temp.append([shift - ppm_span, shift + ppm_span])
        print("testing temp", temp)
        temp = merge_intervals(temp)
        intervals.append(temp)
    return intervals


def merge_all_intervals(shifts_pred_mu, ppm_span=1, pad=None):
    interval_matrix = create_interval_list(shifts_pred_mu, ppm_span, pad)
    intervals = []
    for idx, row in enumerate(interval_matrix):
        for interval in row:
            intervals.append([interval, [idx]])

    print(intervals)
    intervals.sort(key=lambda x: x[0][0])

    merged = []
    for interval in intervals:
        # print(interval)
        if not merged or merged[-1][0][1] < interval[0][0]:
            merged.append(interval)
        else:
            merged[-1][0][1] = max(merged[-1][0][1], interval[0][1])
            merged[-1][1] = np.concatenate([merged[-1][1], interval[1]])
    return [i for i in reversed(merged)]

def isRingAromatic(mol):
    '''
    returns indicies of all aromatic rings in a molecule.

    mol: rdkit molecule
    '''
    ring_info = mol.GetRingInfo()
    atoms = ring_info.AtomRings()
    aromatic_rings = []
    for i, bondRing in enumerate(ring_info.BondRings()):
        for id in bondRing:
            aromatic = False
            if mol.GetBondWithIdx(id).GetIsAromatic():
                aromatic = True
            else:
                aromatic = False
        if aromatic == True:
            aromatic_rings.append(list(atoms[i]))
    return aromatic_rings

def find_integer_sequences(list):
    '''
    Obtain integer sequences from the queried molecule. Based on how the molecule is built from SMILES
    the integer sequence will represent a spin system.

    list: list of indicies
    '''
    my_sequences = []
    for idx,item in enumerate(list):
        if not idx or item-1 != my_sequences[-1][-1]:
            my_sequences.append([item])
        else:
            my_sequences[-1].append(item)
    return my_sequences



def get_spin_systems(mol):
    '''
    Find the indicies of spin systems from a molecule

    mol: rdkit molecule
    '''
    aroCH1 = Chem.MolFromSmarts('[c&H1]')
    CH3 = Chem.MolFromSmarts('[CH3]')
    CH2 = Chem.MolFromSmarts('[CH2]')
    CH1 = Chem.MolFromSmarts('[CH1]')

    atom_indices = []
    for fragment in [CH3, CH2, CH1]:
        fragment_search = list(mol.GetSubstructMatches(fragment))
        if fragment_search: #ie if the list is not empty
            atom_indices.append(fragment_search)
    atom_indices = [item for sublist in flatten(atom_indices) for item in sublist]
    aro_atom_indicies = list(mol.GetSubstructMatches(aroCH1))
    aro_atom_indicies = [item for sublist in aro_atom_indicies for item in sublist]

    atom_indices = find_integer_sequences(atom_indices)
    aro_atom_indicies = find_integer_sequences(aro_atom_indicies)
    return atom_indices + aro_atom_indicies
