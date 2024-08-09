import glob
import itertools
import multiprocessing as mp
import os
import sys
import traceback
from functools import partial

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from lib.nar import seed_res_repr
from lib.pdb import Structure, parser

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

IUPAC = {
    'R': {'A', 'G'},
    'Y': {'C', 'U'},
    'S': {'G', 'C'},
    'W': {'A', 'U'},
    'K': {'G', 'U'},
    'M': {'A', 'C'},
    'B': {'C', 'G', 'U'},
    'D': {'A', 'G', 'U'},
    'H': {'A', 'C', 'U'},
    'V': {'A', 'C', 'G'},
    'N': {'A', 'C', 'G', 'U'},
    **dict(zip(seed_res_repr.keys(),
               map(lambda x: {x},
                   seed_res_repr.keys())))
}


pd.set_option('mode.chained_assignment', None)

rres    = '#1'
qres    = '#1'
rresneg = ''
qresneg = ''

rnosub = 'False'
qnosub = 'False'

rrst    = None
qrst    = None

sizemin     = 1.
sizemax     = 1e10
rmsdmin     = 0.
rmsdmax     = 1e10
rmsdsizemin = 0.
rmsdsizemax = 1e10
resrmsdmin  = 0.
resrmsdmax  = 1e10
matchrange  = 3.


saveto     = ''
saveres    = ''
saveformat = ''

silent = 'False'

if 'fork' in mp.get_all_start_methods():
    mp.set_start_method('fork')
    threads = mp.cpu_count()
else:
    threads = 1

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
    return np.sqrt(np.sum(np.multiply(dX, dX)) / len(X))

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
    
    tab_1 = q_tab.copy()
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

    struct.saveto(folder, q_saveforamt)

def tolist(arg:'str') -> 'list[str]':
    paths = []

    bashls = glob.glob(arg)
    if bashls:
        if len(bashls) == 1 and os.path.isdir(arg):
            for item in os.listdir(arg):
                path = os.path.join(arg, item)
                if os.path.isfile(path):
                    paths.append(path)
        else:
            for path in bashls:
                if os.path.isfile(path):
                    paths.append(path)
    else:
        paths.append(arg)

    return paths


def splitFormat(paths:'list[str]') -> 'list[tuple[str, str]]':
    def fmt(path:'str') -> 'str':
        formatter = {
            'CIF'  : 'CIF',
            'MMCIF': 'CIF',
            'PDB'  : 'PDB',
        }
        _, file   = os.path.split(path)
        name, ext = os.path.splitext(file)
        fmt = formatter.get(ext.lstrip('.').upper(), 'PDB')
        return [path, fmt, name]

    return [*map(fmt, paths)]


class Collection:

    def __init__(self, path, fmt, name,
                 label, res, resneg, neg, seed) -> 'None':
        
        self.path   = path
        self.fmt    = fmt
        self.name   = name
        self.label  = label
        self.res    = res
        self.resneg = resneg
        self.neg    = neg
        self.seed   = seed

        struct = parser(path, fmt, name)
        struct.drop_duplicates_alt_id(keep=keep)
        # resstruct = struct.get_res_substruct(
        #     (res, resneg)[neg],
        #     neg
        # )
        resstruct = struct.get_res_substruct_2(res, resneg)

        data, noise = describe(resstruct)

        if not data:
            # msg = 'No {}={} nucleotides in the {}={} for {}seed={}'.format(
            #     ('{}res'.format(label), '{}resneg'.format(label))[neg],
            #     (res, resneg)[neg],
            #     label,
            #     path,
            #     label,
            #     seed
            # )
            # raise Exception(msg)
            msg = 'No {}res:="{}" - "{}"=:{}resneg nucleotides in the {}={}'.format(
                label,
                res,
                resneg,
                label, 
                label,
                path,
            )
            raise Exception(msg)

        else:
            code, prim, avg, scnd, eval = zip(*data)
            avg = np.vstack(avg)

        seed_code = set(struct.get_res_code(seed))
        if not seed_code:
            # msg = 'No {}seed={} nucleotides in the {}={} for {}={}'.format(
            #     label,
            #     seed,
            #     ('{}res'.format(label), 'rresneg'.format(label))[neg],
            #     (res, resneg)[neg],
            #     label,
            #     path
            # )
            # raise Exception(msg)
            msg = 'No {}seed={} nucleotides in the {}res:="{}" - "{}"=:{}resneg for {}={}'.format(
                label,
                seed,
                label,
                res,
                resneg,
                label,
                label,
                path
            )
            raise Exception(msg)
        else:
            seed_npc  = set(noise) & seed_code
            seed_code = set(code)  & seed_code

        ind = [i for i, code in enumerate(code) if code in seed_code]

        count = len(code)

        cols = [
            'pdbx_PDB_model_num',
            'auth_asym_id',
            'auth_comp_id',
            'auth_seq_id',
            'pdbx_PDB_ins_code'
        ]
        tab = struct.tab

        if '?' in tab[cols[-1]].values:
            groups = (tab
                      .groupby(tab[cols]
                               .replace('?', '')
                               .astype(str)
                               .apply('.'.join, axis=1)))
        else:
            groups = (tab
                      .groupby(tab[cols]
                               .astype(str)
                               .apply('.'.join, axis=1)))

        self.struct   = struct
        self.ind      = ind
        self.count    = count
        self.seed_npc = seed_npc
        self.code     = code
        self.prim     = prim
        self.scnd     = scnd
        self.eval     = eval
        self.avg      = avg
        self.groups   = groups

    def __repr__(self) -> 'str':
        return 'Collection({}path={}, count={})'.format(
            self.label, self.path, self.count
        )


class Reference(Collection):

    def __init__(self, path, fmt, name,
                 res, resneg, neg, seed, label='r') -> 'None':
        super().__init__(path, fmt, name,
                         label,res, resneg, neg, seed)
        self.avg_tree = KDTree(self.avg)


class Query(Collection):

    def __init__(self, path, fmt, name, 
                 res, resneg, neg, seed, label='q') -> 'None':
        super().__init__(path, fmt, name,
                         label, res, resneg, neg, seed)
        self.saveformat = None
        self.tab_1      = None
        self.max_models = None


    def set_save(self, saveformat:'str', saveres:'str'):
        try:
            saveformat = {
                'CIF'  : 'CIF',
                'MMCIF': 'CIF',
                'PDB'  : 'PDB',
                ''     : self.fmt
            }[saveformat.lstrip('.').upper()]
        except KeyError:
            msg = (
                'Invalid saveformat value saveformat={}. '\
                'Acceptable values for saveformat are PDB, CIF or MMCIF'
            ).format(saveformat)
            raise KeyError(msg)
        self.saveformat = saveformat

        if saveres:
            ssmask = self.struct.get_res_mask(saveres)
        else:
            if self.neg:
                ssmask = self.struct.get_res_mask(self.resneg)
                ssmask^= True
            else:
                ssmask = self.struct.get_res_mask(self.res)

        self.tab_1      = self.struct.tab[ssmask]
        self.max_models = self.tab_1['pdbx_PDB_model_num'].max()


def splitNeighborsPairIndexSets(neighbors:'list[tuple]', q_count:'int'):
    rindex = []
    qindex = []

    for v in neighbors:
        pindex = [(i // q_count, i % q_count) for i in v]
        ri, qi = map(set, zip(*pindex))
        rindex.append(ri)
        qindex.append(qi)

    return rindex, qindex


def nosubFilter(tab:'pd.DataFrame',
                 rnosub:'bool', qnosub:'bool',
                 q_count:'int') -> 'pd.DataFrame':

    def nosubs(g:'list[pd.Series[set]]') -> 'list[int]':

        n = len(g)

        if n == 1:
            return g[0].index

        index = []

        for i in range(n - 1):
            sgroup:'pd.Series[set]' = g[i]
            empty = False
            for j in range(i+1, n):
                lgroup:'pd.Series[set]' = g[j]
                for v in lgroup:
                    sgroup = sgroup[1^sgroup.apply(v.issuperset)]
                    if sgroup.empty:
                        empty = True
                        break
                if empty:
                    break
            if not empty:
                index.extend(sgroup.index)
        index.extend(lgroup.index)

        return index

    if not (rnosub or qnosub) or tab.empty:
        return tab

    tab = tab[tab['SIZE'] > 0]

    if tab.empty:
        return tab

    tab['rindex'], tab['qindex'] = splitNeighborsPairIndexSets(
        tab['neighbors'], q_count
    )

    if rnosub:
        _, g = zip(*tab.groupby('SIZE', sort=True)['rindex'])
        tab = tab.loc[nosubs(g)]

    if tab.empty:
        return tab

    if qnosub:
        _, g = zip(*tab.groupby('SIZE', sort=True)['qindex'])
        tab = tab.loc[nosubs(g)]

    del tab['rindex']
    del tab['qindex']

    return tab


def rstQueue(rst:'str'):
    spl = [*map(str.strip, rst.split('|'))]
    if len(spl) == 1:
        spl.append('')
    else:
        spl.pop()

    for i in range(0, len(spl), 2):
        spec = spl[i].split()
        rst  = spl[i + 1]
        if rst.replace('.', '').isnumeric():
            rst = float(rst)

        yield spec, rst


def rstStrand(codeseq:'list', groups, dist:'float'=2) -> 'bool':
    coord = ['Cartn_x', 'Cartn_y', 'Cartn_z']

    seq = [groups
           .get_group(code)
           .set_index('auth_atom_id') 
           for code in codeseq]


    res:'pd.DataFrame' = seq[0]
    if "O3'" in res.index:
        O3 = [res.loc["O3'", coord]]
    else:
        return False

    P = []
    for res in seq[1:-1]:
        if "O3'" in res.index:
            O3.append(res.loc["O3'", coord])
        else:
            return False
        if 'P' in res.index:
            P.append(res.loc["P", coord])
        else:
            return False

    res = seq[-1]
    if 'P' in res.index:
        P.append(res.loc["P", coord])
    else:
        return False

    O3 = np.vstack(O3).astype(float)
    P  = np.vstack(P).astype(float)

    return (np.sqrt(((O3 - P) ** 2).sum(axis=1)) < dist).all()


def rstBaseTypes(codeseq:'list[str]', baseTypes:'str') -> 'bool':
    return all(map(lambda x:x.split('.')[2] in baseTypes, codeseq))


def rrstRMSD(x:'dict', index:'list') -> 'bool':

    X, Y = vstack([[r_scnd[i], q_scnd[v]] for i, v in x.items()])
    rot, tran = get_transform(X, Y)

    X, Y = vstack([[r_scnd[i], q_scnd[x[i]]] for i in index])
    Y = np.dot(Y, rot) + tran

    return RMSD(X, Y)


def qrstRMSD(x:'dict', index:'list') -> 'bool':

    X, Y = vstack([[r_scnd[v], q_scnd[i]] for i, v in x.items()])
    rot, tran = get_transform(X, Y)

    X, Y = vstack([[r_scnd[x[i]], q_scnd[i]] for i in index])
    Y = np.dot(Y, rot) + tran

    return RMSD(X, Y)


def rstFilter(tab:'pd.DataFrame',
              rrst:'str|None', qrst:'str|None',
              r:'Reference', q:'Query') -> 'pd.DataFrame':


    if not (rrst or qrst):
        return tab

    tab = tab[tab['SIZE'] > 0]

    if tab.empty:
        return tab

    r2q = tab['neighbors'].apply(
        lambda h: dict([(v // q_count, v % q_count) for v in h])
    )

    if rrst:
        rCodeToIndex = dict(zip(r.code, itertools.count()))
        qIndexToCode = dict(zip(itertools.count(), q.code))

        for spec, rst in rstQueue(rrst):
            index = []
            for subspec in spec:
                for code in r.struct.get_res_code(subspec):
                    ind = rCodeToIndex.get(code, None)
                    if ind is None:
                        return tab[0:0]
                    index.append(ind)
            indexs = set(index)

            r2q = r2q[r2q.apply(lambda x: indexs.issubset(x))]
            if r2q.empty:
                return tab[0:0]

            if rst == '':
                continue

            elif rst == 'strand':
                qCode = r2q.apply(lambda x: [qIndexToCode[x[i]] for i in index])
                r2q = r2q[qCode.apply(lambda x: rstStrand(x, q.groups))]

            elif isinstance(rst, float):
                r2q = r2q[
                    r2q.apply(
                        lambda x:
                            rrstRMSD(x, index)
                    ) <= rst
                ]

            elif rst in IUPAC:
                baseTypes = IUPAC[rst]
                qCode = r2q.apply(lambda x: [qIndexToCode[x[i]] for i in index])
                r2q = r2q[qCode.apply(lambda x: rstBaseTypes(x, baseTypes))]

            else:
                print("WARNING: Ignoring unrecognized restraint: {}"
                      .format(rst), file=sys.stderr)

            if r2q.empty:
                return tab[0:0]

    tab = tab.loc[r2q.index]

    q2r = r2q.apply(lambda x: {v: i for i, v in x.items()})
    if qrst:
        qCodeToIndex = dict(zip(q.code, itertools.count()))
        rIndexToCode = dict(zip(itertools.count(), r.code))

        for spec, rst in rstQueue(qrst):
            index = []
            for subspec in spec:
                for code in q.struct.get_res_code(subspec):
                    ind = qCodeToIndex.get(code, None)
                    if ind is None:
                        return tab[0:0]
                    index.append(ind)

            indexs = set(index)
            q2r = q2r[q2r.apply(lambda x: indexs.issubset(x))]
            if q2r.empty:
                return tab[0:0]

            if rst == '':
                continue

            elif rst == 'strand':
                rCode = q2r.apply(lambda x: [rIndexToCode[x[i]] for i in index])
                q2r = q2r[rCode.apply(lambda x: rstStrand(x, r.groups))]

            elif isinstance(rst, float):
                q2r = q2r[
                    q2r.apply(
                        lambda x:
                            qrstRMSD(x, index)
                    ) <= rst
                ]

            elif rst in IUPAC:
                baseTypes = IUPAC[rst]
                rCode = q2r.apply(lambda x: [rIndexToCode[x[i]] for i in index])
                q2r = q2r[rCode.apply(lambda x: rstBaseTypes(x, baseTypes))]

            else:
                print("WARNING: Ignoring unrecognized restraint: {}"
                      .format(rst), file=sys.stderr)

            if r2q.empty:
                return tab[0:0]

    tab = tab.loc[q2r.index]

    return tab


if  __name__ == '__main__':

    argv = sys.argv[1:]

    if any([arg in help_args for arg in argv]):
        with open(ROOT_DIR + '/help.txt', 'r') as helper:
            print(helper.read())
        exit()
    else:
        i = 0
        while i < len(argv):
            k = argv[i]
            if k.startswith('-'):
                k = k.lstrip('-')
                v = '='.join([k, argv[i+1]])
                argv[i] = v
                del argv[i+1]
                i += 1
                continue
            i += 1

        kwargs = dict([arg.split('=') for arg in argv])

    threads = int(kwargs.get('threads', threads))

    if threads != 1:
        if threads <= 0:
            threads = mp.cpu_count()
        else:
            threads = min(threads, mp.cpu_count())
        delta = 15 * threads

    sizemin     = float(kwargs.get('sizemin', sizemin))
    sizemax     = float(kwargs.get('sizemax', sizemax))

    resrmsdmin  = float(kwargs.get('resrmsdmin', resrmsdmin))
    resrmsdmax  = float(kwargs.get('resrmsdmax', resrmsdmax))

    rmsdmin     = float(kwargs.get('rmsdmin', rmsdmin))
    rmsdmax     = float(kwargs.get('rmsdmax', rmsdmax))

    rmsdsizemin = float(kwargs.get('rmsdsizemin', rmsdsizemin))
    rmsdsizemax = float(kwargs.get('rmsdsizemax', rmsdsizemax))

    matchrange  = float(kwargs.get('matchrange', matchrange))

    rnosub     = eval(kwargs.get('rnosub', rnosub).capitalize())
    qnosub     = eval(kwargs.get('qnosub', qnosub).capitalize())

    rrst        = kwargs.get('rrst', rrst)
    qrst        = kwargs.get('qrst', qrst)

    if 'trim' in kwargs:
        trim = kwargs['trim'].lower()
        if trim == '0' or trim == 'false':
            trim = False
        else:
            trim = True
    else:
        trim = False

    if trim:
        sizemin = max(sizemin, 0)

    rres        = kwargs.get('rres', rres)
    rresneg     = kwargs.get('rresneg', rresneg)
    rneg        = bool(rresneg)
    rseed       = kwargs.get('rseed', '#')

    qres        = kwargs.get('qres', qres)
    qresneg     = kwargs.get('qresneg', qresneg)
    qneg        = bool(qresneg)
    qseed       = kwargs.get('qseed', '#')

    saveto      = kwargs.get('saveto',  saveto)
    saveres     = kwargs.get('saveres', saveres)
    saveformat  = kwargs.get('saveformat', saveformat)

    silent      = eval(kwargs.get('silent', silent).capitalize())

    try:
        rformat = {
            'CIF'  : 'CIF',
            'MMCIF': 'CIF',
            'PDB'  : 'PDB',
            ''     : ''
        }[kwargs.get('rformat', '').lstrip('.').upper()]
    except KeyError:
        raise KeyError("Incorrect value rformat={}".format(kwargs
                                                           .get('rformat')))
    rlist       = sorted(splitFormat(tolist(kwargs.get('r'))))
    if rformat:
        for item in rlist:
            item[1] = rformat

    try:
        qformat = {
            'CIF'  : 'CIF',
            'MMCIF': 'CIF',
            'PDB'  : 'PDB',
            ''     : ''
        }[kwargs.get('qformat', '').lstrip('.').upper()]
    except KeyError:
        raise KeyError("Incorrect value qformat={}".format(kwargs
                                                           .get('qformat')))
    qlist = sorted(splitFormat(tolist(kwargs.get('q'))))
    if qformat:
        for item in qlist:
            item[1] = qformat

    reference = partial(Reference,
                        res=rres,
                        resneg=rresneg,
                        neg=rneg,
                        seed=rseed)
    query     = partial(Query,
                        res=qres,
                        resneg=qresneg,
                        neg=qneg,
                        seed=qseed)

    save_state = False
    if saveto:
        if len(rlist) > 1:
            save_state = True

            for path, _, _ in rlist:
                _, file = os.path.split(path)
                folder = os.path.join(saveto, file)
                os.makedirs(folder, exist_ok=True)

            else:
                os.makedirs(saveto, exist_ok=True)

    extended_columns =  len(rlist) > 1 or len(qlist) > 1

    if extended_columns:
        columns = ['neighbors', 'SIZE', 'RMSD', 'RMSDSIZE', 'RESRMSD',
                   'PRIM', 'SCND', 'REF', 'QRY']
        print('ID',*columns[1:], sep='\t')

    else:
        columns = ['neighbors', 'SIZE', 'RMSD', 'RMSDSIZE', 'RESRMSD',
                   'PRIM', 'SCND']
        print('ID', *columns[1:], sep='\t')


    for i in rlist:

        try:

            r:'Reference' = reference(*i)
            r_code      = r.code
            r_prim      = r.prim
            r_scnd      = r.scnd
            r_avg_tree  = r.avg_tree
            r_eval      = r.eval
            rstruct     = r.struct

            if save_state:
                _, file = os.path.split(r.path)
                folder = os.path.join(saveto, file)
            else:
                folder = saveto

        except Exception as e:
            if silent:
                print('Exception: r={}'.format(r.path), file=sys.stderr)
                tb = e.__traceback__
                print(''.join(traceback.format_tb(tb)), file=sys.stderr)
                continue
            else:
                raise e

        for j in qlist:
            try:
                q:'Query' = query(*j)
                if saveto:
                    q.set_save(saveformat, saveres)
                    q_tab = q.tab_1
                q_code  = q.code
                q_prim  = q.prim
                q_scnd  = q.scnd
                q_avg   = q.avg
                q_eval  = q.eval
                q_count = q.count
                qstruct = q.struct
                q_saveforamt = q.saveformat

                result     = {}
                indx_pairs = itertools.product(r.ind, q.ind)

                if threads == 1:
                    for p, out in ((i, artem(*i)) for i in indx_pairs):
                        if out:
                            m, n = p 
                            if out in result:
                                result[out].append(m*q_count + n)
                            else:
                                result[out] = [m*q_count + n]
                else:
                    cnt     = 0
                    cnt_max = len(r.ind) * len(q.ind)
                    pool    = mp.Pool(threads)
                    while cnt < cnt_max:
                        inp = [next(indx_pairs)
                                for _ in range(min(cnt_max - cnt, delta))]
                        for p, out in zip(inp, pool.starmap(artem, inp)):
                            if out:
                                m, n = p
                                if out in result:
                                    result[out].append(m*q.count + n)
                                else:
                                    result[out] = [m*q.count + n]
                        cnt += delta

                items = result.items()
                del result

                tabrows = []

                if sizemin <= 0:
                    for npc in r.seed_npc:
                        tabrows.append((tuple(), 0, None, None, None, npc, None))
                    for npc in q.seed_npc:
                        tabrows.append((tuple(), 0, None, None, None, None, npc))

                for k, v in items:
                    size     = len(k) - 2
                    rmsd     = k[-2]
                    resrmsd  = k[-1]
                    rmsdsize = rmsd / size

                    prim = ','.join(
                        [
                            '='.join([r_code[s // q_count],
                                    q_code[s % q_count]])
                            for s in v
                        ]
                    )
                    scnd = ','.join(
                        [
                            '='.join([r_code[s // q_count],
                                    q_code[s % q_count]])
                            for s in k[:-2]
                        ]
                    )
                    tabrows.append((k[:-2], size, rmsd, rmsdsize,
                                    resrmsd, prim, scnd))

                tab = pd.DataFrame(tabrows,
                                    columns=columns[:-2] if extended_columns 
                                    else columns)
                tab.sort_values(
                    ['SIZE', 'RMSDSIZE'], 
                    ascending=[True, False], 
                    inplace=True
                )

                tab = nosubFilter(tab, rnosub, qnosub, q_count)
                tab = rstFilter(tab, rrst, qrst, r, q)

                tab.index = list(range(1, len(tab) + 1))
                tab.index.name = 'ID'

                if extended_columns:
                    tab['REF'] = r.path
                    tab['QRY'] = q.path

                tab.to_csv(
                    sys.stdout,
                    columns=columns[1:],
                    sep='\t',
                    float_format='{:0.3f}'.format,
                    header=False
                )

                if not saveto:
                    continue

                if threads == 1:
                    for superimpose in tab[tab['SIZE'] > 0].iloc:
                        save_superimpose(superimpose)
                else:
                    pool.map(save_superimpose, tab[tab['SIZE'] > 0].iloc)
                    pool.close()

            except Exception as e:
                if silent:
                    print('Exception: r={}, q={}'.format(r.path, q.path), file=sys.stderr)
                    tb = e.__traceback__
                    print(''.join(traceback.format_tb(tb)), file=sys.stderr)
                    continue
                else:
                    raise e
