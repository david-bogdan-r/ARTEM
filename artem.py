import sys
import os
import itertools
import gc

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
rresneg = ''
qresneg = ''
rseed   = ''
qseed   = ''

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

# Keep alternative atom locations: 'first', 'last', False
keep = 'last'

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
seed_res_repr = nar.join_res_repr(seed_res_repr)


def get_transform(r:'np.ndarray', q:'np.ndarray'):
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


def RMSD(r:'np.ndarray', q:'np.ndarray') -> 'float':
    diff = r - q
    return np.sqrt(np.sum(np.sum(np.multiply(diff, diff))) / len(r))


def apply_transform(coord:'np.ndarray', rotran):
    rot, tran = rotran
    return np.dot(coord, rot) + tran


def mutual_nb(dist) -> 'list':
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


def saver(item:'pd.Series') -> None:
    struct = sstruct.apply_transform(item['TRAN'])
    struct.rename('{}_{}'.format(struct, item.name))
    struct.saveto(saveto, saveformat)



if  __name__ == '__main__':
    # Processing inputs
    
    kwargs = dict([arg.split('=') for arg in sys.argv[1:]])
    
    threads = int(kwargs.get('threads', threads))
    if threads != 1:
        # Multiprocessing is available only for UNIX-like systems
        mp.set_start_method('fork')
        
        if threads < 0:
            threads = mp.cpu_count()
        else:
            threads = min(threads, mp.cpu_count())
    
    r       = kwargs.get('r')
    rres    = kwargs.get('rres', rres)
    rresneg = kwargs.get('rresneg', rresneg)
    rseed   = kwargs.get('rseed', rseed)
    rformat = kwargs.get('rformat', rformat)
    
    q       = kwargs.get('q')
    qres    = kwargs.get('qres', qres)
    qresneg = kwargs.get('qresneg', qresneg)
    qseed   = kwargs.get('qseed', qseed)
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
    
    available_format = {'PDB', 'CIF'}
    
    rext = rext.upper()
    if rext in available_format:
        rformat = rext
    
    qext = qext.upper()
    if qext in available_format:
        qformat = qext
    
    saveformat = kwargs.get('saveformat', qformat)
    
    
    # Model preprocessing
    
    rstruct  = pdb.parser(r, rformat, rname)
    rstruct.drop_duplicates_alt_id(keep=keep)
    rnegcase = bool(rresneg)
    rsstruct = rstruct.get_res_substruct(
        [rres, rresneg][rnegcase],
        rnegcase
    )
    rrres, rures = rsstruct.artem_desc(seed_res_repr)
    if not rrres:
        msg = 'No {}={} nucleotides in the {} for seed'.format(
            ['rres', 'rresneg'][rnegcase],
            [rres, rresneg][rnegcase],
            r
        )
        raise ValueError(msg)
    rseed_code = rsstruct.get_res_code(rseed)
    if not rseed_code:
        msg = 'No rseed={} nucleotides in the {}={} for seed {}'.format(
            rseed,
            ['rres', 'rresneg'][rnegcase],
            [rres, rresneg][rnegcase],
            r
        )
        raise ValueError(msg)
    
    
    r_code, r_prim, r_avg, r_scnd, r_eval = zip(*rrres)
    r_avg = np.vstack(r_avg)
    rseed_code = set(r_code) & set(rseed_code)
    r_ind = [i for i, code in enumerate(r_code) if code in rseed_code]
    
    
    qstruct  = pdb.parser(q, qformat, qname)
    qstruct.drop_duplicates_alt_id(keep=keep)
    qnegcase = bool(qresneg)
    qsstruct = qstruct.get_res_substruct(
        [qres, qresneg],
        qnegcase
    )
    qrres, qures = qsstruct.artem_desc(seed_res_repr)
    if not qrres:
        msg = 'No {}={} nucleotides in the {} for seed'.format(
            ['qres', 'qresneg'][qnegcase],
            [qres, qresneg][qnegcase],
            q
        )
        raise ValueError(msg)
    qseed_code = qsstruct.get_res_code(qseed)
    if not qseed_code:
        msg = 'No qseed={} nucleotides in the {}={} for seed {}'.format(
            qseed,
            ['qres', 'qresneg'][qnegcase],
            [qres, qresneg][qnegcase],
            q
        )
        raise ValueError(msg)
    
    q_code, q_prim, q_avg, q_scnd, q_eval = zip(*qrres)
    q_avg = np.vstack(q_avg)
    qseed_code = set(q_code) & set(qseed_code)
    q_ind = [i for i, code in enumerate(q_code) if code in qseed_code]
    
    indx_pairs = list(itertools.product(r_ind, q_ind))
    
    r_avg_tree = KDTree(r_avg)
    if threads == 1:
        result = [task(m, n) for m, n in indx_pairs]
    else:
        pool = mp.Pool(threads)
        
        delta   = 15 * threads
        cnt     = 0
        cnt_max = len(indx_pairs)
        result  = []
        while cnt < cnt_max:
            result.extend(
                pool.starmap(task, indx_pairs[cnt:cnt + delta])
            )
            cnt += delta
    gc.collect()
    
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
        seeds_code = []
        for ind in seeds:
            i, j = indx_pairs[ind]
            seeds_code.append('='.join([r_code[i], q_code[j]]))
        seeds_code = ','.join(seeds_code)
        
        match_code = []
        for i, j in item[-3]:
            match_code.append('='.join([r_code[i], q_code[j]]))
        match_code = ','.join(match_code)
        tab.append([*item[:3], seeds_code, match_code, item[4]])
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
        if saveres:
            sstruct = qstruct.get_res_substruct(saveres)
        else:
            sstruct = qsstruct
        if threads != 1:
            pool = mp.Pool(threads)
            pool.map(saver, tab[tab['SIZE'] > 0].iloc)
        else:
            for item in tab[tab['SIZE'] > 0].iloc:
                saver(item)