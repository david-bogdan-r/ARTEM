import sys
import os
import itertools


import pandas as pd
import numpy  as np
import multiprocessing as mp

from lib import nar
from lib import pdb
from scipy.spatial import KDTree

rres = ''
qres = ''

saveto  = None
saveres = None

rmsdmin = 0.
rmsdmax = 1e10

sizemin = 0.
sizemax = 1e10

rmsdsizemin = 0.
rmsdsizemax = 1e10

matchrange = 3.

rformat = 'PDB'
qformat = 'PDB'

saveformat = 'PDB'

threads = 1

repr_struct_res = (
    nar.five_atom_repr,     # for primary alignment
    nar.five_atom_repr,     # to calculate centers of mass
    nar.three_atom_repr,    # for secondary alignment
    nar.three_atom_repr     # to calculate the RMSD
)


# func

def get_rotran(r:np.ndarray, q:np.ndarray):
    r_avg = r.mean(axis=0)
    q_avg = q.mean(axis=0)
    
    r = r - r_avg
    q = q - q_avg
    
    M = np.dot(np.transpose(q), r)
    u, s, vh = np.linalg.svd(M)
    
    rot = np.transpose(np.dot(np.transpose(vh), np.transpose(u)))
    if np.linalg.det(rot) < 0:
        vh[2] = -vh[2]
        rot = np.transpose(np.dot(np.transpose(vh), np.transpose(u)))
    tran = r_avg - np.dot(q_avg, rot)
    
    return rot, tran


def RMSD_simpl(r:np.ndarraym, q:np.ndarray) -> float:
    diff = r - q
    return np.sum(np.sum(np.multiply(diff, diff)))


def apply_tran(coord:np.ndarray, rotran):
    rot, tran = rotran
    return np.dot(coord, rot) + tran


def mutual_nb(r:np.ndarray, q:np.ndarray):
    pass


if  __name__ == '__main__':
    parser = pdb.Parser()
    
    kwargs = dict([arg.split('=') for arg in sys.argv[1:]])
    
    r = kwargs.get('r')
    q = kwargs.get('q')
    
    rformat = kwargs.get('rformat', rformat)
    qformat = kwargs.get('qformat', qformat)
    
    rname, rext = r.split(os.sep)[-1].split('.')
    qname, qext = q.split(os.sep)[-1].split('.')
    
    rext = rext.upper()
    qext = qext.upper()
    
    if rext in ['PDB', 'CIF']:
        rformat = rext
    
    if qext in ['PDB', 'CIF']:
        qformat = qext
    
    rstruct = parser(r, rformat, rname)
    qstruct = parser(q, qformat, qname)
    
    rres = kwargs.get('rres', rres)
    qres = kwargs.get('qres', qres)
    
    rsstruct = rstruct.get_sub_struct(rres)
    qsstruct = qstruct.get_sub_struct(qres)
    
    rsstruct.drop_duplicates_alt_id(keep='last')
    qsstruct.drop_duplicates_alt_id(keep='last')
    
    # pre proc repr_struct_res
    prep = []
    repr_struct = {}
    for repr_res in repr_struct_res:
        if repr_res in prep:
            for k in repr_res:
                if k in repr_struct:
                    repr_struct[k].append(repr_res[k])
                else:
                    repr_struct[k] = [repr_res[k]]
        else:
            for k in repr_res.keys():
                repr_res[k] = [pd.Index(v.split()) for v in repr_res[k]]
                
                if k in repr_struct:
                    repr_struct[k].append(repr_res[k])
                else:
                    repr_struct[k] = [repr_res[k]]
            
            prep.append(repr_res)
    #
    
    rdata = rsstruct.artem_desc(repr_struct)
    qdata = qsstruct.artem_desc(repr_struct)