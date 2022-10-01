import sys
import os
import itertools

import pandas as pd
import numpy  as np
import multiprocessing as mp

from lib import nar
from lib import pdb
from scipy.spatial import KDTree

rres    = ''
qres    = ''
rformat = 'PDB'
qformat = 'PDB'

rmsdmin     = 0.
rmsdmax     = 1e10
sizemin     = 0.
sizemax     = 1e10
rmsdsizemin = 0.
rmsdsizemax = 1e10
matchrange  = 3.

saveto     = None
saveres    = None
saveformat = 'PDB'

threads = 1

repr_struct_res = (
    nar.five_atom_repr,     # for primary alignment
    nar.five_atom_repr,     # to calculate centers of mass
    nar.three_atom_repr,    # for secondary alignment
    nar.three_atom_repr     # to calculate the RMSD
)


# func

def get_transform(r:np.ndarray, q:np.ndarray):
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


def RMSD(r:np.ndarray, q:np.ndarray) -> float:
    diff = r - q
    return np.sqrt(np.sum(np.sum(np.multiply(diff, diff))) / len(r))


def apply_transform(coord:np.ndarray, rotran):
    rot, tran = rotran
    return np.dot(coord, rot) + tran


def mutual_nb(dist) -> list:
    U_1 = {}
    U_2 = {}
    
    for e in dist:
        v_1, v_2, d = e
        if v_1 not in U_1:
            U_1[v_1] = v_2, d
        else:
            if d < U_1[v_1][1]:
                U_1[v_1] = v_2, d
        
        if v_2 not in U_2:
            U_2[v_2] = v_1, d
        else:
            if d < U_2[v_2][1]:
                U_2[v_2] = v_1, d
    
    alt = []
    for v_2 in U_2:
        v_1, d = U_2[v_2]
        if U_1[v_1][0] == v_2:
            alt.append((v_1, v_2))
    
    return alt


def vstack(alt):
    ref_coord, alg_coord = zip(*alt)
    return np.vstack(ref_coord), np.vstack(alg_coord)


if  __name__ == '__main__':
    kwargs = dict([arg.split('=') for arg in sys.argv[1:]])
    
    r       = kwargs.get('r')
    q       = kwargs.get('q')
    rres    = kwargs.get('rres', rres)
    qres    = kwargs.get('qres', qres)
    rformat = kwargs.get('rformat', rformat)
    qformat = kwargs.get('qformat', qformat)
    
    rmsdmin     = float(kwargs.get('rmsdmin', rmsdmin))
    rmsdmax     = float(kwargs.get('rmsdmax', rmsdmax))
    sizemin     = float(kwargs.get('sizemin', sizemin))
    sizemax     = float(kwargs.get('sizemax', sizemax))
    rmsdsizemin = float(kwargs.get('rmsdsizemin', rmsdsizemin))
    rmsdsizemax = float(kwargs.get('rmsdsizemax', rmsdsizemax))
    matchrange  = float(kwargs.get('matchrange', matchrange))
    
    saveto     = kwargs.get('saveto', saveto)
    saveres    = kwargs.get('saveres', saveres)
    
    threads = int(kwargs.get('threads', threads))
    
    rname, rext = r.split(os.sep)[-1].split('.')
    qname, qext = q.split(os.sep)[-1].split('.')
    
    rext = rext.upper()
    qext = qext.upper()
    
    if rext in ['PDB', 'CIF']:
        rformat = rext
    
    if qext in ['PDB', 'CIF']:
        qformat = qext
    
    saveformat = kwargs.get('saveformat', qformat)
    
    rstruct = pdb.parser(r, rformat, rname)
    qstruct = pdb.parser(q, qformat, qname)
    
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
    
    r_code, r_prim, r_avg, r_scnd, r_rmsd = rsstruct.artem_desc(repr_struct)
    q_code, q_prim, q_avg, q_scnd, q_rmsd = qsstruct.artem_desc(repr_struct)
    
    code_queue = list(itertools.product(r_code, q_code))
    
    # Primary alignment
    r_coord, q_coord = zip(*itertools.product(r_prim, q_prim))
    transform = map(get_transform, r_coord, q_coord)
    
    # Calculation of mutual neighbour's
    q_avg = map(apply_transform, itertools.repeat(q_avg), transform)
    
    r_tree = KDTree(r_avg)
    q_tree = map(KDTree, q_avg)
    dist   = map(
        lambda x:
            r_tree.sparse_distance_matrix(
                x,
                matchrange,
                p=2,
                output_type='ndarray'
            ),
        q_tree
    )
    nb = list(map(mutual_nb, dist))
    
    # Secondary alignment
    nb_scnd = [[(r_scnd[i], q_scnd[j]) for i, j in m] 
                for m in nb]
    r_coord, q_coord = zip(*map(vstack, nb_scnd))
    transform = list(map(get_transform, r_coord, q_coord))
    
    # RMSD calculate
    nb_rmsd = [[(r_rmsd[i], q_rmsd[j]) for i, j in m] 
                for m in nb]
    r_coord, q_coord = zip(*map(vstack, nb_rmsd))
    q_coord = map(apply_transform, q_coord, transform)
    rmsd    = map(RMSD, r_coord, q_coord)

    
    ans   = []
    count = itertools.count()
    size  = map(len, nb)
    for c, s, r in zip(count, size, rmsd):
        if not sizemin <= s <= sizemax:
            continue
        if not rmsdmin <= r <= rmsdmax:
            continue
        if not rmsdsizemin <= r / s <= rmsdsizemax:
            continue
        
        ans.append((c, s, r, r / s))
    ans.sort(key=lambda x: x[3])
    
    c = 1
    for i, s, r, rs in ans:
        seed_pair = code_queue[i]
        alt = sorted(nb[i], key=lambda u: u[0])
        r_ind, q_ind = zip(*alt)
        
        r_scode = [r_code[k] for k in r_ind]
        q_scode = [q_code[k] for k in q_ind]
        
        print('{}\t{}={}\t{}\t{:0.3f}\t{:0.3f}'.format(c, *seed_pair, s, r, rs))
        print('\t'.join(['{}={}'.format(rc, qc) for rc, qc in zip(r_scode, q_scode)]))
        
        c += 1
    
    if saveto:
        if saveres:
            sstruct = qstruct.get_sub_struct(saveres)
        else:
            sstruct = qsstruct
        
        c = 1
        for i in [a[0] for a in ans]:
            struct = sstruct.apply_transform(transform[i])
            struct.rename('{}_{}'.format(struct, c))
            struct.saveto(saveto, saveformat)
            
            c += 1