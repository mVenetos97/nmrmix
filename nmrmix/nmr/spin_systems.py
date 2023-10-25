from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.io.xyz import XYZ
from pymatgen.analysis.local_env import CovalentBondNN
from itertools import groupby
from pymatgen.core.structure import Molecule

def coupling_dfs(node, graph, mol, visited, equiv_protons, site_subset):
    if node not in visited:
        visited.add(node)
        temp = []
        for neighbor in graph[node]:
            if mol.species[neighbor].symbol == 'H':
                temp.append(neighbor)
        if len(temp) > 0:
            site_subset.append(mol[node])
            idx = str(node)
            equiv_protons[idx] = temp
            for neighbor in graph[node]:
                if mol.species[neighbor].symbol != 'C':
                    site_subset.append(mol[neighbor])
                elif mol.species[neighbor].symbol == 'C':
                    coupling_dfs(neighbor, graph, mol, visited, equiv_protons, site_subset)
    return equiv_protons, site_subset

def are_equivalent_spin_systems(spin_sys_1, spin_sys_2):
    strat = CovalentBondNN()
    graph_1 = MoleculeGraph.with_local_env_strategy(spin_sys_1, strat)
    graph_2 = MoleculeGraph.with_local_env_strategy(spin_sys_2, strat)
    return graph_1.isomorphic_to(graph_2)

def is_valid_coupling_path(edge, mol, spin_system):
    atom = [i.symbol for i in mol.species]
    if atom[edge[0]] == 'C' and atom[edge[1]] == 'C':
        for sub_system in spin_system:
            keys = sub_system.keys()
            if str(edge[0]) in keys and str(edge[1]) in keys:
                return True #return a spin system object
    
    return False