'''
    phyiscal_constants.py
        Set the thresholds, interactions, and otherwise hardcoded data from the files here.

        If you're wondering: "Why is this hardcoded? What kind of over caffienated
            code monkeys wrote this hacky research code?"

        It's hardcoded because this is research code and these values aren't
            going to be updated ever
'''

import numpy as np

'''
    map ascii names to Van Der Waals Radii in 10^(-10)m.
'''
vdW_radii = dict({
    "Isoalloxazine": {
        'N1':		1.82,
        'C2':		1.91,
        'O2':		1.66,
        'N3':		1.82,
        'C4':		1.91,
        'O4':		1.66,
        'C4X':		1.91,
        'N5':		1.82,
        'C5X':		1.91,
        'C6':		1.91,
        'C7':		1.91,
        'C7M':  	1.91,
        'C8':		1.91,
        'C8M':		1.91,
        'C9':		1.91,
        'C9A':	    1.91,
        'N10':	    1.82,
        'C10':		1.91,
    },
        'HOH': 1.72,
        'N':        1.82,
        'C':		1.91,
        'O':		1.66,
        'CA':	    1.91,
        'CB':	    1.91,
        'CGx':		1.91,
        'CDx':		1.91,
        'CEx':		1.91,
        'CZx':		1.91,
        'CH2':		1.91,
        'SD':		2.00,
        'OGx':		1.72,
        'SG':		2.00,
        'ODx':		1.66,
        'OEx':		1.66,
        'NDx':		1.82,
        'NEx':		1.82,
        'OH':		1.72,
        'NZ':	    1.88,
        'NHx':		1.82,
        'OXT':		1.66,
        'SE':		np.nan,

})

"""
    :param atom_name accepts the radius of the atom's vdW in Angstroms (10^-10)
    :param is_water bool defaults to false
        returns the oxygen vdW in water when set to true, ignores atom name input
        returns the oxygen vdW in other chains otherwise

    returns the radius of atom
"""
def get_vdW_radius(atom_name):
    if atom_name in vdW_radii:
        return vdW_radii[atom_name]
    elif atom_name in vdW_radii['Isoalloxazine']:
        return vdW_radii['Isoalloxazine'][atom_name]
    else:
        print ("Warning: atom_name not found in vdW table: ", atom_name)
        return np.inf

"""
    max or min combination of two atoms experiencing van der Waal's forces
    calculated by
            max/min vdw_radii +/- artificial error
    Artificial error constant added in order to avoid issues in measurements
    and floating point error.

    *** NOTE: depricated and hacky - only use for prototyping

"""
# vdW_bounds = dict({
#     'upper': max(max(vdW_radii.values()), max(vdW_radii['Isoalloxazine'].values())) * 2 + 0.01,
#     'lower': min(min(vdW_radii.values()), min(vdW_radii['Isoalloxazine'].values())) * 2 + 0.01,
# })
vdW_bounds = dict({
    'upper': 2 * 2 + 0.01,
    'lower': 1.66 * 2 + 0.01,
})

#FIXME - update what PDB speak is
"""
    These map ASCII names of abreviations of molecules to
    a number representing somekind of chemical properties in PDB speak.

    Atoms that don't have an number are represented by 'NaN'.
"""
chemical_codes = dict({
        'N':    1,
        'C':    2,
        'O':    3,
        'CA':   2,
        'CB':   2,
        'CGx':  2,
        'CDx':  2,
        'CEx':  2,
        'CZx':  2,
        'CH2':  2,
        'SD':   4,
        'OGx':  5,
        'SG':   6,
        'ODx':  7,
        'OEx':  7,
        'NDx':  1,
        'NEx':	np.nan,
        'OH':   np.nan,
        'NZ':   np.nan,
        'NHx':  np.nan,
        'OXT':  np.nan,
        'SE':   np.nan
})
