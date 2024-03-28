from pandas import Index

_5_atomic_representation = {
    'A':   ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),
    'G':   ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),

    '1MA': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),
    '2MG': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),
    '6MZ': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),
    '7MG': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),
    'A2M': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),
    'MA6': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),
    'OMG': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),
    'YYG': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),
    'SAM': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),

    'C':   ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),
    'U':   ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),

    '4AC': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),
    '4SU': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),
    '5MC': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),
    '5MU': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),
    'LV2': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),
    'OMC': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),
    'OMU': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),
    'SSU': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),
    'UR3': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),

    'PSU': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'C5', 'C4', 'C2'),
    'B8N': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'C5', 'C4', 'C2'),
    '3TD': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'C5', 'C4', 'C2'),
    'UY1': ('P', "C1' C2' O2' C3' O3' C4' O4' C5' O5'", 'C5', 'C4', 'C2'),

    'DA' : ('P', "C1' C2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),
    'DG' : ('P', "C1' C2' C3' O3' C4' O4' C5' O5'", 'N9', 'C2', 'C6'),
    'DC' : ('P', "C1' C2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),
    'DT' : ('P', "C1' C2' C3' O3' C4' O4' C5' O5'", 'N1', 'C2', 'C4'),

    "CDP": ("PA", "C1' C2' O2' C3' O3' C4' O4' C5' O5'", "N1", "C2", "C4"),
    "UDP": ("PA", "C1' C2' O2' C3' O3' C4' O4' C5' O5'", "N1", "C2", "C4"),
    "ADP": ("PA", "C1' C2' O2' C3' O3' C4' O4' C5' O5'", "N9", "C2", "C6"),
    "GDP": ("PA", "C1' C2' O2' C3' O3' C4' O4' C5' O5'", "N9", "C2", "C6"),

    "CTP": ("PA", "C1' C2' O2' C3' O3' C4' O4' C5' O5'", "N1", "C2", "C4"),
    "UTP": ("PA", "C1' C2' O2' C3' O3' C4' O4' C5' O5'", "N1", "C2", "C4"),
    "ATP": ("PA", "C1' C2' O2' C3' O3' C4' O4' C5' O5'", "N9", "C2", "C6"),
    "GTP": ("PA", "C1' C2' O2' C3' O3' C4' O4' C5' O5'", "N9", "C2", "C6")
}

_3_atomic_representation = {
    'A':   ('N9', 'C2', 'C6'),
    'G':   ('N9', 'C2', 'C6'),

    '1MA': ('N9', 'C2', 'C6'),
    '2MG': ('N9', 'C2', 'C6'),
    '6MZ': ('N9', 'C2', 'C6'),
    '7MG': ('N9', 'C2', 'C6'),
    'A2M': ('N9', 'C2', 'C6'),
    'MA6': ('N9', 'C2', 'C6'),
    'OMG': ('N9', 'C2', 'C6'),
    'YYG': ('N9', 'C2', 'C6'),
    'SAM': ('N9', 'C2', 'C6'),

    'C':   ('N1', 'C2', 'C4'),
    'U':   ('N1', 'C2', 'C4'),

    '4AC': ('N1', 'C2', 'C4'),
    '4SU': ('N1', 'C2', 'C4'),
    '5MC': ('N1', 'C2', 'C4'),
    '5MU': ('N1', 'C2', 'C4'),
    'LV2': ('N1', 'C2', 'C4'),
    'OMC': ('N1', 'C2', 'C4'),
    'OMU': ('N1', 'C2', 'C4'),
    'SSU': ('N1', 'C2', 'C4'),
    'UR3': ('N1', 'C2', 'C4'),

    'PSU': ('C5', 'C4', 'C2'),
    'B8N': ('C5', 'C4', 'C2'),
    '3TD': ('C5', 'C4', 'C2'),
    'UY1': ('C5', 'C4', 'C2'),

    'DA' : ('N9', 'C2', 'C6'),
    'DG' : ('N9', 'C2', 'C6'),
    'DC' : ('N1', 'C2', 'C4'),
    'DT' : ('N1', 'C2', 'C4'),
    
    "CDP": ('N1', 'C2', 'C4'),
    "UDP": ('N1', 'C2', 'C4'),
    "ADP": ('N9', 'C2', 'C6'),
    "GDP": ('N9', 'C2', 'C6'),

    "CTP": ('N1', 'C2', 'C4'),
    "UTP": ('N1', 'C2', 'C4'),
    "ATP": ('N9', 'C2', 'C6'),
    "GTP": ('N9', 'C2', 'C6')
}

seed_res_repr = (
    _5_atomic_representation, # For primary alignment
    _5_atomic_representation, # To calculate centers of mass
    _3_atomic_representation, # For secondary alignment
    _3_atomic_representation, # To calculate RMSD
)

prepared = []
for res_repr in seed_res_repr:
    if res_repr in prepared:
        continue
    for res, atoms in res_repr.items():
        res_repr[res] = [Index(atom.split()) for atom in atoms]
    prepared.append(res_repr)

carrier = set.intersection(
    *map(
        lambda x: set(x.keys()),
        seed_res_repr
    )
)

seed_res_repr_ = seed_res_repr
seed_res_repr  = {}
for res in carrier:
    seed_res_repr[res] = [
        srr[res] for srr in seed_res_repr_
    ]
