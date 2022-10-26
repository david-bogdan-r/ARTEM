import os
import gc
import sys
import itertools
import psutil
import time

import pandas as pd
import numpy  as np
import multiprocessing as mp

from scipy.spatial import KDTree
from shutil import rmtree
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

saveto     = ''
saveres    = ''

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
seed_res_repr = nar.join_res_repr(seed_res_repr)

# Keep alternative atom locations: 'first', 'last', False
keep = 'last'

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


def mutual_nb(dist:'list') -> 'list':
    '''
    Input:
    dist - list of threes [rres_intID, qres_intID, distance]
    
    Output:
    alt - list of pairs of mutually nearest residues (rres_intID, qres_intID)
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
    
    if saveto:
        sup_sstract = sstruct.apply_transform(transform)
        sup_sstract.rename('{}_{}_{}'.format(sup_sstract, m, n))
        sup_sstract.saveto('tmp', saveformat)
    
    nb = tuple(sorted(nb))
    return (nb, rmsd)



if  __name__ == '__main__':
    # Processing inputs
    argv = sys.argv[1:]
    
    if argv[0] in {'--H', '-H', '--h', '-h', '--help', '-help'}:
        with open('help.txt', 'r') as help:
            print(*help)
        exit(0)
    else:
        kwargs = dict([arg.split('=') for arg in argv])
    
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
    
    
    rname, rext = r.split(os.sep)[-1].split('.')
    qname, qext = q.split(os.sep)[-1].split('.')
    
    available_format = {'PDB', 'CIF'}
    
    rext = rext.upper()
    if rext in available_format:
        rformat = rext
    
    qext = qext.upper()
    if qext in available_format:
        qformat = qext
    
    
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
        [qres, qresneg][qnegcase],
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
    q_count = len(q_code)
    
    
    # Preparing a saved structure
    saveto = kwargs.get('saveto', saveto)
    if saveto:
        os.makedirs(saveto, exist_ok=True)
        saveres    = kwargs.get('saveres', saveres)
        saveformat = kwargs.get('saveformat', qformat).upper()
    
        if saveformat not in available_format:
            if saveformat == 'MMCIF':
                saveformat = 'CIF'
            else:
                msg = 'Invalid saveformat value. Acceptable values for saveformat'
                msg+= 'are PDB, CIF or MMCIF (case-insensitive)'
                raise TypeError(msg)
        
        if saveres:
            sstruct = qstruct.get_res_substruct(saveres)
        else:
            sstruct = qsstruct
        
        tmp_path = 'tmp/'    + sstruct.name + '_{}_{}.' + saveformat.lower()
        svt_path = saveto + '/' + sstruct.name + '_{}.' + saveformat.lower()
    
    
    # ARTEM Computations 
    
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
            print(cnt, round(cnt / len(indx_pairs), 3), psutil.Process().memory_info().rss, time.time())
        else:
            print(cnt, round(cnt / len(indx_pairs), 3), psutil.Process().memory_info().rss, time.time())
    
    # Output
    
    rows = {}
    for i, rslt in enumerate(result):
        if rslt:
            nb, rmsd = rslt
        else:
            continue
        
        if nb in rows:
            row = rows[nb]
            if rmsd < row[0]:
                row[0] = rmsd
                row.insert(1, i)
            else:
                row.append(i)
        else:
            rows[nb] = [rmsd, i]
    rows = [[i, *v] for i, v in rows.items()]
    
    tabrows = []
    for i, row in enumerate(rows):
        nb   = row[0]
        rmsd = row[1]
        size = len(nb)
        rmsdsize = rmsd / size
        file_id  = row[2]
        seed_id  = row[2:]
        
        prim = ','.join(
            [
                '='.join([r_code[s // q_count], q_code[s % q_count]])
                for s in seed_id
            ]
        )
        
        scnd = ','.join(
            [
                '='.join([r_code[m], q_code[n]])
                for m, n in nb
            ]
        )
        
        tabrows.append((file_id, size, rmsd, rmsdsize, prim, scnd))
    
    
    columns = ['ID', 'SIZE', 'RMSD', 'RMSDSIZE', 'PRIM', 'SCND']
    tab = pd.DataFrame(tabrows, columns=columns)
    tab.sort_values(
        ['SIZE', 'RMSDSIZE'], 
        ascending=[True, False], 
        inplace=True
    )
    tab.index = list(range(1, len(tab) + 1))
    tab.index.name = 'ID'
    rnm = dict(zip(tab['ID'], tab.index))
    
    if saveto:
        for k, v in list(rnm.items()):
            os.rename(tmp_path.format(*indx_pairs[k]), svt_path.format(v))
        rmtree('tmp')
    
    # tab.to_csv(
    #     sys.stdout,
    #     columns=['SIZE', 'RMSD', 'RMSDSIZE', 'PRIM', 'SCND'],
    #     sep='\t',
    #     float_format='{:0.3f}'.format,
    # )