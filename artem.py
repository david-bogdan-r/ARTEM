import os
import sys
import itertools

import pandas as pd
import numpy  as np
import multiprocessing as mp

from scipy.spatial import KDTree
from lib import nar
from lib import pdb


# ignore SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)


# Defaults

rres    = ''
qres    = ''
rresneg = ''
qresneg = ''

rmsdmin     = 0.
rmsdmax     = 1e10
sizemin     = 1.
sizemax     = 1e10
rmsdsizemin = 0.
rmsdsizemax = 1e10
matchrange  = 3.

saveto  = ''
saveres = ''

threads = 1

seed_res_repr = (
    nar.five_atom_repr,     # For primary alignment
    nar.five_atom_repr,     # To calculate centers of mass
    nar.three_atom_repr,    # For secondary alignment
    nar.three_atom_repr,    # To calculate the RMSD
)
seed_res_repr = nar.join_res_repr(seed_res_repr)


keep = 'last'   # keep alternative atom locations: 'first', 'last', False


# Functions

def get_transform(X:'np.ndarray', Y:'np.ndarray'):
    '''
    Returns a linear operator minimizing the RMSD between matrix q and r.
    
    Input:
        r - N x 3 np.ndarray matrix
        q - N x 3 np.ndarray matrix
    Output:
        rot  - 3 x 3 rotation np.ndarray matrix
        tran - 3 - np.ndarray vector 
    '''
    X_avg = X.mean(axis=0)
    Y_avg = Y.mean(axis=0)
    
    X = X - X_avg
    Y = Y - Y_avg
    
    M = np.dot(np.transpose(Y), X)
    S, V, D = np.linalg.svd(M)
    
    A = np.transpose(np.dot(np.transpose(D), np.transpose(S)))
    if np.linalg.det(A) < 0:
        D[2] = -D[2]
        A = np.transpose(np.dot(np.transpose(D), np.transpose(S)))
    B = X_avg - np.dot(Y_avg, A)
    
    return A, B

def apply_transform(X:'np.ndarray', transform):
    A, B = transform
    return np.dot(X, A) + B

def RMSD(X:'np.ndarray', Y:'np.ndarray') -> 'float':
    dX = X - Y
    return np.sqrt(np.sum(np.sum(np.multiply(dX, dX))) / len(X))

def mutual_nearest_neighbors(distances:'list') -> 'list':
    neighbor_1 = {}
    neighbor_2 = {}
    for edge in distances:
        vert_1, vert_2, dist = edge
        if vert_1 not in neighbor_1:
            neighbor_1[vert_1] = vert_2, dist
        else:
            if dist < neighbor_1[vert_1][1]:
                neighbor_1[vert_1] = vert_2, dist
        
        if vert_2 not in neighbor_2:
            neighbor_2[vert_2] = vert_1, dist
        else:
            if dist < neighbor_2[vert_2][1]:
                neighbor_2[vert_2] = vert_1, dist
    
    neighbors = []
    for vert_2 in neighbor_2:
        vert_1, dist = neighbor_2[vert_2]
        if neighbor_1[vert_1][0] == vert_2:
            neighbors.append((vert_1, vert_2))
    return neighbors


def vstack(ndarray_pairs):
    X, Y = zip(*ndarray_pairs)
    return np.vstack(X), np.vstack(Y)

def describe(struct: 'pdb.Structure'):
    tab  = struct.get_tab().copy()
    tab.set_index('auth_atom_id', inplace=True)
    
    code_mask = struct._get_code_mask().set_axis(tab.index)
    
    data  = []
    noise = []
    cartn_cols = ['Cartn_x', 'Cartn_y', 'Cartn_z']
    for code, res_tab in tab[cartn_cols].groupby(code_mask, sort=False):
        res_id = code.split('.', 3)[-2]
        if res_id not in seed_res_repr.keys():
            noise.append(code)
            continue
        
        flg = False
        res_data = []
        for res_repr in seed_res_repr[res_id]:
            res_cartn = []
            for atom_repr in res_repr:
                try:
                    atom_cartn = res_tab.loc[atom_repr].values
                except:
                    noise.append(code)
                    flg = True
                    break
                
                if len(atom_cartn) > 1:
                    atom_cartn = atom_cartn.mean(axis=0)
                
                res_cartn.append(atom_cartn)
            
            if flg:
                break
            
            res_cartn = np.vstack(res_cartn)
            res_data.append(res_cartn)
        
        if flg:
            continue
        
        res_data[1] = res_data[1].mean(axis=0)
        data.append([code, *res_data])
    
    return data, noise

def artem(m, n):
    prim_transform  = get_transform(r_prim[m], q_prim[n])
    
    q_avg_tree = KDTree(apply_transform(q_avg, prim_transform))
    dist = r_avg_tree.sparse_distance_matrix(
        q_avg_tree,
        matchrange,
        p=2,
        output_type='ndarray'
    )

    neighbors = mutual_nearest_neighbors(dist)
    size = len(neighbors)
    if not sizemin <= size <= sizemax:
        return None
    
    X, Y = vstack([[r_scnd[i], q_scnd[j]] for i, j in neighbors])
    scnd_transform = get_transform(X, Y)
    
    X, Y = vstack([[r_eval[i], q_eval[j]] for i, j in neighbors])
    
    rmsd = RMSD(X, apply_transform(Y, scnd_transform))
    if not rmsdmin <= rmsd <= rmsdmax:
        return None
    
    rmsdsize = rmsd / size
    if not rmsdsizemin <= rmsdsize <= rmsdsizemax:
        return None
    
    neighbors = tuple(sorted(neighbors))
    return (neighbors, rmsd)


def save_superimpose(superimpose:'pd.Series') -> 'None':
    neighbors = superimpose['neighbors']
    X, Y = vstack([[r_scnd[i], q_scnd[j]] for i, j in neighbors])
    rot, tran = get_transform(X, Y)
    
    coord_cols = ['Cartn_x', 'Cartn_y', 'Cartn_z']
    
    tab_1 = qstruct.tab[ssmask]
    tab_1.loc[:, coord_cols] = np.round(
        np.dot(tab_1[coord_cols], rot) + tran, 
        3
    )
    max_models = tab_1['pdbx_PDB_model_num'].max()
    
    rsavecode, qsavecode = zip(*neighbors)
    
    rsavecode = [r_code[i] for i in rsavecode]
    rmask = rstruct._get_code_mask().isin(rsavecode)
    tab_2 = rstruct.tab[rmask]
    tab_2['pdbx_PDB_model_num'] = max_models + 1
    
    qsavecode = [q_code[i] for i in qsavecode]
    qmask = qstruct._get_code_mask().isin(qsavecode)
    tab_3 = qstruct.tab[qmask]
    tab_3.loc[:, coord_cols] = np.round(
        np.dot(tab_3[coord_cols], rot) + tran, 
        3
    )
    tab_3['pdbx_PDB_model_num'] = max_models + 2
    
    struct = pdb.Structure('{}_{}'.format(qstruct, superimpose.name))
    used_cols = min([tab_1.columns, tab_2.columns], key=len)
    tab = pd.concat([tab_1[used_cols], tab_2[used_cols], tab_3[used_cols]])
    
    if rstruct.fmt != qstruct.fmt:
        tab['label_alt_id'].replace('.', '', inplace=True)
        tab['pdbx_PDB_ins_code'].replace('?', '', inplace=True)
        tab['pdbx_formal_charge'].replace('?', '', inplace=True)
        
        struct.set_tab(tab)
        struct.set_fmt('PDB')
    else:
        struct.set_tab(tab)
        struct.set_fmt(qstruct.fmt)
    
    struct.saveto(saveto, saveformat)


if  __name__ == '__main__':
    
    # Processing inputs
    
    argv = sys.argv[1:]
    if argv[0] in {'--H', '-H', '--h', '-h', '--help', '-help'}:
        with open('README.md', 'r') as rdme:
            print(*rdme)
        exit()
    else:
        kwargs = dict([arg.split('=') for arg in argv])
    
    threads = int(kwargs.get('threads', threads))
    if threads != 1:
        mp.set_start_method('fork')     # ARTEM multiprocessing is available only for UNIX-like systems
        if threads <= 0:
            threads = mp.cpu_count()
        else:
            threads = min(threads, mp.cpu_count())
    
    r       = kwargs.get('r')
    rres    = kwargs.get('rres', rres)
    
    rresneg = kwargs.get('rresneg', rresneg)
    rneg    = bool(rresneg)
    rseed   = kwargs.get('rseed', '#')
    
    rformat = kwargs.get('rformat', None)
    rname, rext = r.split(os.sep)[-1].split('.')
    rext = rext.upper()
    if not rformat:
        if rext in pdb.formats:
            rformat = rext
        else:
            rformat = 'PDB'
    else:
        rformat = rformat.upper()
    
    q       = kwargs.get('q')
    qres    = kwargs.get('qres', qres)
    qresneg = kwargs.get('qresneg', qresneg)
    qneg    = bool(qresneg)
    qseed   = kwargs.get('qseed', '#')
    qformat = kwargs.get('qformat', None)
    qname, qext = q.split(os.sep)[-1].split('.')
    qext = qext.upper()
    if not qformat:
        if qext in pdb.formats:
            qformat = qext
        else:
            qformat = 'PDB'
    else:
        qformat = qformat.upper()
    
    sizemin     = float(kwargs.get('sizemin', sizemin))
    sizemax     = float(kwargs.get('sizemax', sizemax))
    
    rmsdmin     = float(kwargs.get('rmsdmin', rmsdmin))
    rmsdmax     = float(kwargs.get('rmsdmax', rmsdmax))
    
    rmsdsizemin = float(kwargs.get('rmsdsizemin', rmsdsizemin))
    rmsdsizemax = float(kwargs.get('rmsdsizemax', rmsdsizemax))
    
    matchrange  = float(kwargs.get('matchrange', matchrange))
    
    
    # Model preprocessing
    
    rstruct  = pdb.parser(r, rformat, rname)
    rstruct.drop_duplicates_alt_id(keep=keep)
    
    rresstruct = rstruct.get_res_substruct(
        (rres, rresneg)[rneg],
        rneg
    )
    
    rdata, rnoise = describe(rresstruct)
    if not rdata:
        msg = 'No {}={} nucleotides in the r={} for rseed={}'.format(
            ('rres', 'rresneg')[rneg],
            (rres, rresneg)[rneg],
            r,
            rseed
        )
        raise Exception(msg)
    else:
        r_code, r_prim, r_avg, r_scnd, r_eval = zip(*rdata)
        r_avg = np.vstack(r_avg)
    
    rseed_code = set(rstruct.get_res_code(rseed))
    if not rseed_code:
        msg = 'No rseed={} nucleotides in the {}={} for r={}'.format(
            rseed,
            ('rres', 'rresneg')[rneg],
            (rres, rresneg)[rneg],
            r
        )
        raise Exception(msg)
    else:
        rseed_npc  = set(rnoise) & rseed_code
        rseed_code = set(r_code) & rseed_code
    
    r_ind = [i for i, code in enumerate(r_code) if code in rseed_code]
    
    
    qstruct  = pdb.parser(q, qformat, qname)
    qstruct.drop_duplicates_alt_id(keep=keep)
    
    qneg = bool(qresneg)
    qresstruct = qstruct.get_res_substruct(
        (qres, qresneg)[qneg],
        qneg
    )
    qrres, qures = describe(qresstruct)
    if not qrres:
        msg = 'No {}={} nucleotides in the q={} for qseed={}'.format(
            ('qres', 'qresneg')[qneg],
            (qres, qresneg)[qneg],
            q,
            qseed
        )
        raise Exception(msg)
    else:
        q_code, q_prim, q_avg, q_scnd, q_eval = zip(*qrres)
        q_avg = np.vstack(q_avg)
    
    qseed_code = set(qstruct.get_res_code(qseed))
    if not qseed_code:
        msg = 'No qseed={} nucleotides in the {}={} for q={}'.format(
            qseed,
            ('qres', 'qresneg')[qneg],
            (qres, qresneg)[qneg],
            q
        )
        raise Exception(msg)
    else:
        qseed_npc  = set(qures)  & qseed_code
        qseed_code = set(q_code) & qseed_code
    
    q_ind = [i for i, code in enumerate(q_code) if code in qseed_code]
    q_count = len(q_code)
    
    
    # Preparing a saved structure
    saveto = kwargs.get('saveto', saveto)
    if saveto:
        os.makedirs(saveto, exist_ok=True)
        saveres    = kwargs.get('saveres', saveres)
        saveformat = kwargs.get('saveformat', qformat).upper()
        
        if saveformat not in pdb.formats:
            if saveformat == 'MMCIF':
                saveformat = 'CIF'
            else:
                msg = '''Invalid saveformat value
                \rAcceptable values for saveformat are PDB, CIF or MMCIF'''
                raise TypeError(msg)
        
        if saveres:
            ssmask = qstruct.get_res_mask(saveres)
        else:
            ssmask = qresstruct.tab['id'].astype(bool)
    
    
    # ARTEM Computations 
    rows = {}
    indx_pairs = list(itertools.product(r_ind, q_ind))
    r_avg_tree = KDTree(r_avg)
    if threads == 1:
        i = 0 
        for rslt in (artem(m, n) for m, n in indx_pairs):
            if rslt:
                nb, rmsd = rslt
            else:
                i += 1
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
            i += 1 
    else:
        pool = mp.Pool(threads)
        
        delta   = 15 * threads
        cnt     = 0
        cnt_max = len(indx_pairs)
        while cnt < cnt_max:
            i = cnt 
            for rslt in pool.starmap(artem, indx_pairs[cnt:cnt + delta]):
                if rslt:
                    nb, rmsd = rslt
                else:
                    i += 1
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
                i += 1 
            cnt += delta
    rows = [[i, *v] for i, v in rows.items()]
    
    # Output
    
    tabrows = []
    if sizemin <= 0:
        for pair in itertools.product(rseed_npc, qseed_npc | qseed_code):
            tabrows.append((None, 0, None, None, '='.join(pair), None))
        for pair in itertools.product(rseed_npc | rseed_code, qseed_npc):
            tabrows.append((None, 0, None, None, '='.join(pair), None))
        
    for i, row in enumerate(rows):
        nb   = row[0]
        size = len(nb)
        rmsd = row[1]
        rmsdsize = rmsd / size
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
        
        tabrows.append((nb, size, rmsd, rmsdsize, prim, scnd))
    
    
    columns = ['neighbors', 'SIZE', 'RMSD', 'RMSDSIZE', 'PRIM', 'SCND']
    tab = pd.DataFrame(tabrows, columns=columns)
    tab.sort_values(
        ['SIZE', 'RMSDSIZE'], 
        ascending=[True, False], 
        inplace=True
    )
    tab.index = list(range(1, len(tab) + 1))
    tab.index.name = 'ID'
    
    tab.to_csv(
        sys.stdout,
        columns=['SIZE', 'RMSD', 'RMSDSIZE', 'PRIM', 'SCND'],
        sep='\t',
        float_format='{:0.3f}'.format,
    )
     
    if saveto:
        if 'pool' in dir():
            pool.map(save_superimpose, tab[tab['SIZE'] > 0].iloc)
        else:
            for superimpose in tab[tab['SIZE'] > 0].iloc:
                save_superimpose(superimpose)