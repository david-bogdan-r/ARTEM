import sys
import os
import itertools

import pandas as pd
import numpy  as np
import multiprocessing as mp

from scipy.spatial import KDTree
from lib import nar
from lib import pdb

# ignore SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)


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

seed_res_repr = (
    # For primary alignment
    nar.five_atom_repr,
    
    # To calculate centers of mass
    nar.five_atom_repr,
    
    # For secondary alignment
    nar.three_atom_repr,
    
    # To calculate the RMSD
    nar.three_atom_repr,
)

# Keep alternative atom locations: 'first', 'last', False
keep = 'last'


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
    '''
    dist: list of [rresId1, qresId2, distance]
    '''
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


def task(m, n):
    transform  = get_transform(r_prim[m], q_prim[n])
    
    q_avg_tree = KDTree(apply_transform(q_avg, transform))
    dist = r_avg_tree.sparse_distance_matrix(
        q_avg_tree,
        matchrange,
        p=2,
        output_type='ndarray'
    )
    
    nb   = mutual_nb(dist)
    size = len(nb)
    if not sizemin <= size <= sizemax:
        return None
    
    scnd = vstack([[r_scnd[i], q_scnd[j]] for i, j in nb])
    transform  = get_transform(*scnd)
    
    r_coord, q_coord = vstack([[r_eval[i], q_eval[j]] for i, j in nb])
    q_coord = apply_transform(q_coord, transform)
    
    rmsd = RMSD(r_coord, q_coord)
    if not rmsdmin <= rmsd <= rmsdmax:
        return None
    
    rmsdsize = rmsd / size
    if not rmsdsizemin <= rmsdsize <= rmsdsizemax:
        return None
    
    nb.sort()
    return [size, rmsd, rmsdsize, tuple(nb), transform]


def saver(item:pd.Series) -> None:
    struct = sstruct.apply_transform(item['TRAN'])
    struct.rename('{}_{}'.format(struct, item.name))
    struct.saveto(saveto, saveformat)



if  __name__ == '__main__':
    kwargs = dict([arg.split('=') for arg in sys.argv[1:]])
    
    threads = int(kwargs.get('threads', threads))
    if threads != 1:
        try:
            mp.set_start_method('fork')
        except:
            print(
                'Multiprocessing is available only for UNIX systems', 
                file=sys.stderr
            )
            threads = 1
    
    r       = kwargs.get('r')
    q       = kwargs.get('q')
    rres    = kwargs.get('rres', rres)
    qres    = kwargs.get('qres', qres)
    rformat = kwargs.get('rformat', rformat)
    qformat = kwargs.get('qformat', qformat)
    
    sizemin     = float(kwargs.get('sizemin', sizemin))
    sizemax     = float(kwargs.get('sizemax', sizemax))
    rmsdmin     = float(kwargs.get('rmsdmin', rmsdmin))
    rmsdmax     = float(kwargs.get('rmsdmax', rmsdmax))
    rmsdsizemin = float(kwargs.get('rmsdsizemin', rmsdsizemin))
    rmsdsizemax = float(kwargs.get('rmsdsizemax', rmsdsizemax))
    matchrange  = float(kwargs.get('matchrange', matchrange))
    
    saveto     = kwargs.get('saveto', saveto)
    saveres    = kwargs.get('saveres', saveres)
    
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
    
    rstruct.drop_duplicates_alt_id(keep=keep)
    qstruct.drop_duplicates_alt_id(keep=keep)
    
    rsstruct = rstruct.get_sub_struct(rres)
    qsstruct = qstruct.get_sub_struct(qres)
    
    if saveto:
        if saveres:
            sstruct = qstruct.get_sub_struct(saveres)
        else:
            sstruct = qsstruct
    
    carrier = set.intersection(
        *map(
            lambda x: set(x.keys()),
            seed_res_repr
        )
    )
    res_repr = {}
    for res in carrier:
        res_repr[res] = [rr[res] for rr in  seed_res_repr]
    
    rrres, rures = rsstruct.artem_desc(res_repr)
    qrres, qures = qsstruct.artem_desc(res_repr)
    
    if not rrres or not qrres:
        columns = ['SIZE', 'RMSD', 'RMSDSIZE', 'PRIM', 'SCND', 'TRAN']
        tab     = []
        for code in rures:
            tab.append([0, None, None, rsstruct.name, code, None])
        for code in qures:
            tab.append([0, None, None, qsstruct.name, code, None])
        tab = pd.DataFrame(tab, columns=columns)
        tab.index = range(1, len(tab) + 1)
        tab.index.name = 'ID'
        tab.to_csv(
            sys.stdout,
            columns=['SIZE', 'RMSD', 'RMSDSIZE', 'PRIM', 'SCND'],
            sep='\t',
            float_format='{:0.3f}'.format
        )
        exit()
    
    r_code, r_prim, r_avg, r_scnd, r_eval = zip(*rrres)
    q_code, q_prim, q_avg, q_scnd, q_eval = zip(*qrres)
    
    r_avg = np.vstack(r_avg)
    q_avg = np.vstack(q_avg)
    
    cpairs = list(itertools.product(r_code, q_code))
    ipairs = itertools.product(range(len(r_code)), range(len(q_code)))
    
    r_avg_tree = KDTree(r_avg)
    if threads == 1:
        result = [task(m, n) for m, n in ipairs]
    else:
        if threads == -1:
            pool = mp.Pool(mp.cpu_count())
        else:
            pool = mp.Pool(min(threads, mp.cpu_count()))
        result = pool.starmap(task, ipairs)
    
    items = {}
    for i, item in enumerate(result):
        if item:
            nb = item[-2]
            if nb not in items:
                items[nb] = item + [[i]]
            else:
                items[nb][-1].append(i)
                
                # selecting the minimum RMSD
                if item[1] < items[nb][1]:
                    items[nb][1] = item[1]
                    items[nb][2] = item[2]
    
    columns = ['SIZE', 'RMSD', 'RMSDSIZE', 'PRIM', 'SCND', 'TRAN']
    tab     = []
    for code in rures:
        tab.append([0, None, None, rsstruct.name, code, None])
    for code in qures:
        tab.append([0, None, None, qsstruct.name, code, None])
    
    for item in items.values():
        seeds = item[-1]
        seeds = ','.join(['='.join(cpairs[i]) for i in seeds])
        pairs = item[-3]
        pairs = ','.join('='.join([r_code[m], q_code[n]]) for m, n in pairs)
        tab.append([*item[:3], seeds, pairs, item[4]])
    tab = pd.DataFrame(tab, columns=columns)
    
    tab.sort_values(
        ['SIZE', 'RMSDSIZE'], 
        ascending=[True, False], 
        inplace=True
    )
    tab.index = range(1, len(tab) + 1)
    tab.index.name = 'ID'
    tab.to_csv(
        sys.stdout,
        columns=['SIZE', 'RMSD', 'RMSDSIZE', 'PRIM', 'SCND'],
        sep='\t',
        float_format='{:0.3f}'.format
    )
    
    if saveto:
        if threads != 1:
            pool.map(saver, tab[tab['SIZE'] > 0].iloc)
        else:
            for item in tab[tab['SIZE'] > 0].iloc:
                saver(item)