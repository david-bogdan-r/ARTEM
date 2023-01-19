import os
import sys
import itertools

import pandas as pd
import numpy  as np
import multiprocessing as mp

from scipy.spatial import KDTree

from lib.pdb import Structure, parser
from lib.nar import seed_res_repr

pd.set_option('mode.chained_assignment', None)

rres    = '#1'
qres    = '#1'
rresneg = ''
qresneg = ''

sizemin     = 1.
sizemax     = 1e10
rmsdmin     = 0.
rmsdmax     = 1e10
rmsdsizemin = 0.
rmsdsizemax = 1e10
resrmsdmin  = 0.
resrmsdmax  = 1e10
matchrange  = 3.

saveto  = ''
saveres = ''

threads = 1

trim = False

keep = 'last'

help_args = {'--H', '-H', '--h', '-h', '--help', '-help'}


def get_transform(X:'np.ndarray', Y:'np.ndarray'):
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

def describe(struct: 'Structure'):
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
    transform  = get_transform(r_prim[m], q_prim[n])
    q_avg_tree = KDTree(apply_transform(q_avg, transform))
    dist = r_avg_tree.sparse_distance_matrix(
        q_avg_tree,
        matchrange,
        p=2,
        output_type='ndarray'
    )
    
    neighbors = mutual_nearest_neighbors(dist)
    size      = len(neighbors)
    
    if trim:
        while size >= sizemin:
            X, Y = vstack([[r_scnd[i], q_scnd[j]] for i, j in neighbors])
            transform = get_transform(X, Y)
            
            resrmsd = np.array(
                [
                    RMSD(
                        r_eval[i], 
                        apply_transform(q_eval[j], transform)
                    )
                    for i, j in neighbors
                ]
            )
            argmax  = resrmsd.argmax()
            resrmsd = resrmsd[argmax]
            
            if resrmsd > resrmsdmax:
                neighbors.pop(argmax)
                size -= 1
                continue
            
            X, Y = vstack([[r_eval[i], q_eval[j]] for i, j in neighbors])
            
            rmsd = RMSD(X, apply_transform(Y, transform))
            if rmsd > rmsdmax:
                neighbors.pop(argmax)
                size -= 1
                continue
            
            rmsdsize = rmsd / size 
            if rmsdsize > rmsdsizemax:
                neighbors.pop(argmax)
                size -= 1
                continue
            
            
            if size > sizemax:
                return None
            
            if resrmsd < resrmsdmin:
                return None
            
            if rmsd < rmsdmin:
                return None
            
            if rmsdsize < rmsdsizemin:
                return None
            
            break
        else:
            return None
    
    else:
        if not sizemin <= size <= sizemax:
            return None
        
        X, Y = vstack([[r_scnd[i], q_scnd[j]] for i, j in neighbors])
        transform = get_transform(X, Y)
        
        X, Y = vstack([[r_eval[i], q_eval[j]] for i, j in neighbors])
        
        rmsd = RMSD(X, apply_transform(Y, transform))
        if not rmsdmin <= rmsd <= rmsdmax:
            return None
        
        rmsdsize = rmsd / size
        if not rmsdsizemin <= rmsdsize <= rmsdsizemax:
            return None
        
        resrmsd = max(
            [
                RMSD(
                    r_eval[i], 
                    apply_transform(q_eval[j], transform)
                )
                for i, j in neighbors
            ]
        )
        if not resrmsdmin <= resrmsd <= resrmsdmax:
            return None
    
    neighbors = sorted([i*q_count + j for i, j in neighbors])
    neighbors.extend([round(rmsd, 3), round(resrmsd, 3)])
    
    return tuple(neighbors)


def save_superimpose(superimpose:'pd.Series') -> 'None':
    neighbors = superimpose['neighbors']
    neighbors = [(v // q_count, v % q_count) for v in neighbors]
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
    
    struct = Structure('{}_{}'.format(qstruct, superimpose.name))
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
    
    argv = sys.argv[1:]
    if any([arg in help_args for arg in argv]):
        with open('help.txt', 'r') as helper:
            print(helper.read())
        exit()
    else:
        kwargs = dict([arg.split('=') for arg in argv])
    
    threads = int(kwargs.get('threads', threads))
    
    if threads != 1:
        mp.set_start_method('fork')
        if threads <= 0:
            threads = mp.cpu_count()
        else:
            threads = min(threads, mp.cpu_count())
    
    r       = kwargs.get('r')
    rres    = kwargs.get('rres', rres)
    
    rresneg = kwargs.get('rresneg', rresneg)
    rneg    = bool(rresneg)
    rseed   = kwargs.get('rseed', '#')
    
    rname, rext = r.split(os.sep)[-1].split('.')
    rext = rext.upper()
    rext = 'CIF' if rext == 'MMCIF' else rext
    
    rformat = kwargs.get('rformat', '').upper()
    rformat = 'CIF' if rformat == 'MMCIF' else rformat
    
    if rformat not in {'PDB', 'CIF'}:
        if rext not in {'PDB', 'CIF'}:
            rformat = 'PDB'
        else:
            rformat = rext
    
    
    q       = kwargs.get('q')
    qres    = kwargs.get('qres', qres)
    qresneg = kwargs.get('qresneg', qresneg)
    qneg    = bool(qresneg)
    qseed   = kwargs.get('qseed', '#')
    
    qname, qext = q.split(os.sep)[-1].split('.')
    qext = qext.upper()
    qext = 'CIF' if qext == 'MMCIF' else qext
    
    qformat = kwargs.get('qformat', '').upper()
    qformat = 'CIF' if qformat == 'MMCIF' else qformat
    
    if qformat not in {'PDB', 'CIF'}:
        if qext not in {'PDB', 'CIF'}:
            qformat = 'PDB'
        else:
            qformat = qext
    
    
    sizemin     = float(kwargs.get('sizemin', sizemin))
    sizemax     = float(kwargs.get('sizemax', sizemax))
    
    resrmsdmin  = float(kwargs.get('resrmsdmin', resrmsdmin))
    resrmsdmax  = float(kwargs.get('resrmsdmax', resrmsdmax))
    
    rmsdmin     = float(kwargs.get('rmsdmin', rmsdmin))
    rmsdmax     = float(kwargs.get('rmsdmax', rmsdmax))
    
    rmsdsizemin = float(kwargs.get('rmsdsizemin', rmsdsizemin))
    rmsdsizemax = float(kwargs.get('rmsdsizemax', rmsdsizemax))
    
    matchrange  = float(kwargs.get('matchrange', matchrange))
    
    trim    = bool(kwargs.get('trim', trim))
    if trim:
        sizemin = max(sizemin, 0)
    
    rstruct  = parser(r, rformat, rname)
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
    
    
    qstruct  = parser(q, qformat, qname)
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
    
    saveto = kwargs.get('saveto', saveto)
    if saveto:
        os.makedirs(saveto, exist_ok=True)
        saveres    = kwargs.get('saveres', saveres)
        saveformat = kwargs.get('saveformat', qformat).upper()
        saveformat = 'CIF' if saveformat == 'MMCIF' else saveformat
        if saveformat not in {'PDB', 'CIF'}:
            msg = '''Invalid saveformat value
            \rAcceptable values for saveformat are PDB, CIF or MMCIF'''
            raise TypeError(msg)
        
        if saveres:
            ssmask = qstruct.get_res_mask(saveres)
        else:
            if qneg:
                ssmask = qstruct.get_res_mask(qresneg)
                ssmask ^= True
            else:
                ssmask = qstruct.get_res_mask(qres)
    
    result     = {}
    indx_pairs = itertools.product(r_ind, q_ind)
    r_avg_tree = KDTree(r_avg)
    if threads == 1:
        inp = list(indx_pairs)
        for p, out in zip(inp, (artem(m, n) for m, n in inp)):
            if out:
                m, n = p 
                if out in result:
                    result[out].append(m*q_count + n)
                else:
                    result[out] = [m*q_count + n]
    else:
        pool = mp.Pool(threads)
        
        delta   = 15 * threads
        cnt     = 0
        cnt_max = len(r_ind) * len(q_ind)
        while cnt < cnt_max:
            inp = [next(indx_pairs) for _ in range(min(cnt_max-cnt,delta))]
            for p, out in zip(inp, pool.starmap(artem, inp)):
                if out:
                    m, n = p
                    if out in result:
                        result[out].append(m*q_count + n)
                    else:
                        result[out] = [m*q_count + n]
            cnt += delta
    items = result.items()
    del result
    
    tabrows = []
    
    if sizemin <= 0:
        for npc in rseed_npc:
            tabrows.append((None, 0, None, None, None, npc, None))
        for npc in qseed_npc:
            tabrows.append((None, 0, None, None, None, None, npc))
    
    for k, v in items:
        size = len(k) - 2
        rmsd = k[-2]
        resrmsd = k[-1]
        rmsdsize = rmsd / size

        prim = ','.join(
            [
                '='.join([r_code[s // q_count], q_code[s % q_count]])
                for s in v
            ]
        )
        scnd = ','.join(
            [
                '='.join([r_code[s // q_count], q_code[s % q_count]])
                for s in k[:-2]
            ]
        )
        
        tabrows.append((k[:-2], size, rmsd, rmsdsize, resrmsd, prim, scnd))
    
    columns = ['neighbors', 'SIZE', 'RMSD', 'RMSDSIZE', 'RESRMSD', 'PRIM', 'SCND']
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
        columns=['SIZE', 'RMSD', 'RMSDSIZE', 'RESRMSD', 'PRIM', 'SCND'],
        sep='\t',
        float_format='{:0.3f}'.format,
    )
    
    if saveto:
        if 'pool' in globals():
            pool.map(save_superimpose, tab[tab['SIZE'] > 0].iloc)
        else:
            for superimpose in tab[tab['SIZE'] > 0].iloc:
                save_superimpose(superimpose)
