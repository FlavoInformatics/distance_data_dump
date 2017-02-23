import numpy as np
from biopandas import pdb
import pandas as pd
from FilterInteractions import *


def _make_distance_comparator(origin, tolerance=0.2):
    keys = ["x_coord", "y_coord", "z_coord"]
    vdW_keyatom = get_vdW_radius(origin['atom_name'].values[0])
    coords = origin[keys].values[0]
    x_o, y_o, z_o = coords[0], coords[1], coords[2]

    def distance_comparator(point):
        coords = point[keys]
        x, y, z = coords[0], coords[1], coords[2]
        net_x, net_y, net_z = abs(x - x_o), abs(y - y_o), abs(z - z_o)
        if net_x < vdW_bounds['lower'] and net_y < vdW_bounds['lower'] and net_z < vdW_bounds['lower']:
            distance = np.sqrt(net_x **2 + net_y ** 2 + net_z ** 2)
            expected_weak_iteraction_dist = get_vdW_radius(point['atom_name']) + vdW_keyatom
            if distance < expected_weak_iteraction_dist + tolerance and distance > expected_weak_iteraction_dist - tolerance:
                return distance
        return np.nan

    return distance_comparator

atmnums = []
key_atoms = ['C10', 'O4']

dor = pdb.PandasPDB().fetch_pdb('2dor').df
dor = pd.concat([dor['ATOM'], dor["HETATM"]])

for key in key_atoms:
    for x in dor[dor['atom_name'] == key].atom_number:
        atmnums.append(x)

neighbours = []
for num in atmnums:
    atom_data = dor[dor.atom_number == num]
    func = _make_distance_comparator(atom_data)
    distance_from_atom_df = dor.apply(func, axis=1)
    dor['distance'] = distance_from_atom_df
    valid_atom_indices = dor['distance'].notnull()
    valid_key_atoms = dor[valid_atom_indices].sort_values(by=['distance'], ascending=True)
    neighbours.append(valid_key_atoms)

df = pd.DataFrame()
df['atom_number'] = atmnums
df['neighbors'] = neighbours
df.to_pickle('2dor.pkl')

