import os

import pandas as pd 
import numpy  as np

# pd.to_numeric.__defaults__ = ('ignore',) + pd.to_numeric.__defaults__[1:]
# pd.options.mode.chained_assignment = None

formats = {'PDB', 'CIF'}
URL = 'https://files.rcsb.org/view/{}{}'
class Structure:
    count = 0
    
    def __init__(self, name:'str' = '') -> 'None':
        if not name:
            Structure.count += 1
            name = 'struct_{}'.format(Structure.count)
        
        self.name = name
        self.tab  = None    # DataFrame to store the atom_site table
        self.fmt  = None    # str to store the table format [CIF or PDB]
    
    def __str__(self) -> 'str':
        return self.name
    
    def __repr__(self) -> 'str':
        return '<{} Structure>'.format(self)
    
    
    def rename(self, name:'str') -> 'None':
        self.name = name
    
    
    def set_tab(self, tab:'pd.DataFrame') -> 'None':
        self.tab = tab
    
    def get_tab(self) -> 'pd.DataFrame':
        return self.tab
    
    
    def set_fmt(self, fmt:'str') -> 'None':
        self.fmt = fmt
    
    def get_fmt(self) -> 'str':
        return self.fmt
    
    
    def _one_letter_chain_renaming(tab:'pd.DataFrame'):
        '''
        When converting from CIF to PDB format, the chain labels are translated into a one-letter code
        '''
        
        if not hasattr(Structure, 'chain_labels'):
            from string import ascii_letters, digits
            chain_labels = sorted(ascii_letters) + list(digits)
            Structure.chain_labels = chain_labels
        else:
            chain_labels = Structure.chain_labels
        
        cur_chain_labels = tab['auth_asym_id'].astype(str).unique()
        allowed_labels = sorted(set(chain_labels) - set(cur_chain_labels), 
                                key=lambda x: (x.isdigit(), x.islower(), x))
        
        rnm = {}
        cnt = 0
        for lbl in cur_chain_labels:
            if len(lbl) > 1:
                try:
                    rnm[lbl] = allowed_labels[cnt]
                except IndexError:
                    raise IndexError('One-letter chain renaming is not possible. Number of chains exceeds the limit 62.')
                cnt += 1
        
        msg = []
        if cnt:
            tab = tab.copy()
            tab['auth_asym_id'].replace(rnm, inplace=True)

            for k, v in sorted(rnm.items(), key=lambda x:x[1]):
                rmk = REMARK.format(k, v)
                rmk += ' ' * (80 - len(rmk)) + '\n'
                msg.append(rmk)
        
        return tab, msg
    
    
    def saveto(self, folder:'str', fmt:'str' = None) -> 'None':
        os.makedirs(folder, exist_ok=True)
        tab = self.tab.copy()
        
        if not fmt:
            fmt = self.fmt
        
        path = '{folder}/{name}.{ext}'.format(
            folder = folder,
            name   = self.name,
            ext    = fmt.lower()
        )
        file = open(path, 'w')
        
        if fmt == 'PDB':
            if self.fmt != fmt:
                tab, msg = Structure._one_letter_chain_renaming(tab)
                tab = tab.replace('.', '')
                tab.replace('?', '', inplace=True)
            else:
                msg = []
            
            tab['auth_atom_id'] = tab.apply(atom_id_PDBformat, axis=1)

            if tab['occupancy'].dtype == float:
                tab['occupancy'] = tab['occupancy'].apply(lambda x: '{:.2f}'.format(x))
            if tab['B_iso_or_equiv'].dtype == float:
                tab['B_iso_or_equiv'] = tab['B_iso_or_equiv'].apply(lambda x: '{:.2f}'.format(x))

            text = ''.join(msg)
            for model_num, tt in tab.groupby('pdbx_PDB_model_num', sort=False):
                text += MODEL.format(model_num)
                tt['id'] = range(1, len(tt) + 1)
                chain_count = 0
                for asym_id, ttt in tt.groupby('auth_asym_id', sort=False):
                    ttt['id'] = (ttt['id'] + chain_count) % 1_000_000
                    for item in ttt.iloc:
                        text += ATOM.format(**item)
                    item['id'] += 1
                    text  += TER.format(**item)
                    chain_count += 1
                text += ENDMDL
            file.write(text)
        
        elif fmt == 'CIF':
            if self.fmt != fmt:
                tab = tab.copy()
                tab['pdbx_PDB_ins_code'].replace('', '?', inplace=True)
                tab['pdbx_formal_charge'].replace('', '?', inplace=True)
                tab['label_alt_id'].replace('', '.', inplace=True)
                
                tab['label_atom_id']   = tab['auth_atom_id']
                tab['label_comp_id']   = tab['auth_comp_id']
                tab['label_asym_id']   = tab['auth_asym_id']
                # tab['label_seq_id']    = tab['auth_seq_id']
                
                entity = tab['label_asym_id'].unique()
                entity = dict(zip(entity, range(1, len(entity) + 1)))
                tab['label_entity_id'] = tab['label_asym_id'].replace(entity)
                
                label_seq_id = tab[[
                    'pdbx_PDB_model_num',
                    'auth_asym_id', 
                    'auth_comp_id',
                    'auth_seq_id',
                    'pdbx_PDB_ins_code'
                ]].astype(str).apply(lambda x: '.'.join(x), axis=1)
                res_id = label_seq_id.unique()
                replace = dict(zip(res_id, range(len(res_id))))
                label_seq_id.replace(replace, inplace=True)
                tab['label_seq_id'] = label_seq_id

                tab = tab[
                    [
                        'group_PDB',
                        'id',
                        'type_symbol',
                        'label_atom_id',
                        'label_alt_id',
                        'label_comp_id',
                        'label_asym_id',
                        'label_entity_id',
                        'label_seq_id',
                        'pdbx_PDB_ins_code',
                        'Cartn_x',
                        'Cartn_y',
                        'Cartn_z',
                        'occupancy',
                        'B_iso_or_equiv',
                        'pdbx_formal_charge',
                        'auth_seq_id',
                        'auth_comp_id',
                        'auth_asym_id',
                        'auth_atom_id',
                        'pdbx_PDB_model_num'
                    ]
                ]
            
            
            title = 'data_{}\n'.format(self.name.upper())
            file.write(title)
            
            header  = '# \nloop_\n'
            for col in tab.columns:
                header += '_atom_site.{}\n'.format(col)
            file.write(header)

            colen = tab.apply(lambda x: max(x.astype(str).map(len))) + 1
            line_format = ''
            for col, l in colen.items():
                line_format += '{' + f'{col}:<{l}' + '}'
            body = '\n'.join(map(lambda x: line_format.format(**x[1]),
                                 tab.iterrows()))
            # tab.to_string(file, header=False, index=False)
            file.write(body)
            file.write('\n# \n')
        
        file.close()
    
    
    def apply_transform(self, transform) -> 'Structure':
        cols  = ['Cartn_x', 'Cartn_y', 'Cartn_z']
        tab   = self.tab.copy()
        
        coord = tab[cols].values
        rot, tran = transform
        tab.loc[:, cols] = np.round(np.dot(coord, rot) + tran, 3)
        
        struct = Structure(self.name)
        struct.set_tab(tab)
        struct.set_fmt(self.fmt)
        
        return struct
    
    
    def _res_split(res:'str') -> 'dict':
        spl = {'#': None, '/': None, ':': None}
        gen = iter(res)
        c   = next(gen, False)
        
        while c:
            if c in spl.keys():
                spl[c] = ''
                cc = next(gen, False)
                while cc and cc not in spl.keys():
                    spl[c] += cc
                    cc = next(gen, False)
                else:
                    c = cc
            else:
                c = next(gen, False)
        
        if spl['#']:
            spl['#'] = int(spl['#'])
        else:
            if spl['#'] == None:
                spl['#'] = 1 # Not OK.
        
        if spl[':']:
            rng = spl[':'].split('_')
            if len(rng) == 3:
                rng[1] = int(rng[1])
                rng[2] = int(rng[2])
            elif len(rng) == 2:
                try:
                    rng[1] = int(rng[1])
                except:
                    pass
            spl[':'] = rng
        
        return spl
    
    
    def get_res_mask(self, res:'str') -> 'pd.Series':
        tab = self.tab.copy()
        
        mask = tab['id'].astype(bool) ^ True
        for subres in res.split():
            spl = Structure._res_split(subres)
            
            mod = spl['#']
            if mod:
                mod_mask = tab['pdbx_PDB_model_num'].eq(mod)
            else:
                mod_mask = tab['group_PDB'].astype(bool)
            
            chn = spl['/']
            if chn:
                chn_mask = tab['auth_asym_id'].eq(chn)
            else:
                chn_mask = tab['group_PDB'].astype(bool)
            
            rng  = spl[':']
            if rng:
                case = len(rng)
            else:
                case = 0
            
            # res wo ':'
            if case == 0:
                rng_mask = tab['group_PDB'].astype(bool)
        
            # ':N'
            elif case == 1:
                res = rng[0]
                if res:
                    rng_mask = tab['auth_comp_id'].eq(rng[0])
                else:
                    rng_mask = tab['group_PDB'].astype(bool)
            
            #':_num' | ':_numN' | ':N_num' | ':N1_numN2'
            elif case == 2:
                res, num  = rng
                if type(num) == int:
                    if res:
                        rng_mask = tab['auth_seq_id'].eq(num) \
                            & tab['auth_comp_id'].eq(res)
                    else:
                        rng_mask = tab['auth_seq_id'].eq(num)
                else:
                    # ':N1_numN2'
                    dgt = ''
                    for i, c in enumerate(num):
                        if c.isdigit():
                            dgt += c
                        else:
                            break
                    ins_code = num[i:]
                    num = int(dgt)
                    rng_mask = tab['auth_seq_id'].eq(num) \
                        & tab['pdbx_PDB_ins_code'].eq(ins_code)
                    if res:
                        rng_mask &= tab['auth_comp_id'].eq(res)
            
            # ':_num1_num2' | ':N_num1_num2'
            elif case == 3:
                res, num_1, num_2 = rng
                if res:
                    rng_mask = tab['auth_comp_id'].eq(res) \
                        & tab['auth_seq_id'].between(num_1, num_2)
                else:
                    rng_mask = tab['auth_seq_id'].between(num_1, num_2)
            
            # incorrect res
            else:
                rng_mask = tab['group_PDB'].astype(bool) ^ True
        
            mask |= mod_mask & chn_mask & rng_mask
        
        return mask
    
    
    def get_res_substruct(self, res:'str', neg:'bool' = False) -> 'Structure':
        '''
        res=[#[modelIdInt]][/[asymIdStr]][:[compIdStr][_seqIdInt1[compIdStr|_seqIdInt2]]
        
        Returns a Structure object with residues, which are defined by res string.
        If neg is True, the returned structure contains residues that are not defined by res string.
        '''
        
        tab = self.tab
        msk = self.get_res_mask(res)
        
        if neg:
            msk ^=True
        tab = tab[msk]
        
        structure = Structure(self.name)
        structure.set_tab(tab)
        structure.set_fmt(self.fmt)
        
        return structure

    def get_res_substruct_2(self, res:'str',  resneg:'str') -> 'Structure':
        '''
        return res - resneg
        '''
        
        tab = self.tab
        resmask    = self.get_res_mask(res)
        if resneg:
            resmask ^= self.get_res_mask(resneg)

        tab = tab[resmask]
        
        structure = Structure(self.name)
        structure.set_tab(tab)
        structure.set_fmt(self.fmt)
        
        return structure
    
    
    def _get_code_mask(self) -> 'pd.Series':
        tab = self.tab
        code_mask = tab['pdbx_PDB_model_num'].astype(str)\
            + '.' + tab['auth_asym_id'].astype(str)\
            + '.' + tab['auth_comp_id'].astype(str)\
            + '.' + tab['auth_seq_id'].astype(str)\
            + '.' + tab['pdbx_PDB_ins_code'].astype(str).replace('?', '')
        return code_mask
    
    
    def get_res_code(self, res:'str' = '', neg:'bool' = False) -> 'list':
        code_msk = self._get_code_mask()
        msk = self.get_res_mask(res)
        if neg:
            msk ^= True
        res_code = code_msk[msk].unique().tolist()
        
        return res_code
    
    
    def drop_duplicates_alt_id(self, keep:'str' = 'last') -> 'None':
        self.tab.drop_duplicates(
            [
                'pdbx_PDB_model_num', 
                'auth_asym_id', 
                'auth_comp_id',
                'auth_seq_id',
                'pdbx_PDB_ins_code',
                'auth_atom_id'
            ],
            keep=keep,
            inplace=True
        )
    
    
    def artem_desc(self, seed_res_repr):
        if not hasattr(self, 'code_msk'):
            self.set_code_mask()
        
        tab  = self.tab.set_index('auth_atom_id')
        code_msk = self.code_mask.set_axis(tab.index)
        
        rres = []
        ures = []
        
        Cartn_cols = ['Cartn_x', 'Cartn_y', 'Cartn_z']
        for code, t in tab[Cartn_cols].groupby(code_msk, sort=False):
            res_id = code.split('.', 3)[-2]
            if res_id not in seed_res_repr.keys():
                ures.append(code)
                continue
            
            flg = False
            c   = []
            
            for r in seed_res_repr[res_id]:
                m = []
                for rr in r:
                    try:
                        v = t.loc[rr].values
                    except:
                        ures.append(code)
                        flg = True
                        break
                    
                    if len(v) > 1:
                        v = v.mean(axis=0)
                    
                    m.append(v)
                
                if flg:
                    break
                
                m = np.vstack(m)
                c.append(m)
            
            if flg:
                continue
            
            c[1] = c[1].mean(axis=0)
            rres.append([code, *c])
        
        return rres, ures



def parser(path:'str'='', fmt:'str' = 'PDB', name:'str' = '') -> 'Structure':
    pdb_columns = (
        'group_PDB',
        'id',
        'auth_atom_id',
        'label_alt_id',
        'auth_comp_id',
        'auth_asym_id',
        'auth_seq_id',
        'pdbx_PDB_ins_code',
        'Cartn_x',
        'Cartn_y',
        'Cartn_z',
        'occupancy',
        'B_iso_or_equiv',
        'type_symbol',
        'pdbx_formal_charge',
        
        'pdbx_PDB_model_num',   # extra column for the atomic coordinate table
    )

    if fmt == 'PDB':
        rec_names = {'ATOM  ', 'HETATM'}
        
        cur_model = 1
        items     = []
        
        file = []
        if os.path.isfile(path):
            file = open(path, 'r').readlines()
        else:
            if len(name) == 4:
                import requests
                url = URL.format(name.upper(), '.pdb')
                resp = requests.get(url)
                if resp.status_code == 200:
                    file = resp.text.split('\n')
                else:
                    raise Exception(
                        'Could not obtain {name} structure in {fmt} format from RCSB PDB'.format(
                            name=name, fmt=fmt
                        )
                    )


        for line in file:
            line = line.strip()
            if line.startswith('_'):
                raise Exception(
                    'File {} does not contain {} format data'.format(path, fmt)
                )
            
            rec = line[0:6]
            if rec in rec_names:
                item = [
                    line[0 : 6].strip(), # group_PDB
                    line[6 :11].strip(), # id
                    line[12:16].strip(), # auth_atom_id
                    line[16:17].strip(), # label_alt_id
                    line[17:20].strip(), # auth_comp_id
                    line[20:22].strip(), # auth_asym_id
                    line[22:26].strip(), # auth_seq_id
                    line[26:27].strip(), # pdbx_PDB_ins_code
                    line[30:38].strip(), # Cartn_x
                    line[38:46].strip(), # Cartn_y
                    line[46:54].strip(), # Cartn_z
                    line[54:60].strip(), # occupancy
                    line[60:66].strip(), # B_iso_or_equiv
                    line[76:78].strip(), # type_symbol
                    line[78:80].strip(), # pdbx_formal_charge
                    cur_model            # pdbx_PDB_model_num
                ]
                items.append(item)
            elif rec == 'MODEL ':
                cur_model = int(line.split()[1])

        tab = pd.DataFrame(items, columns=pdb_columns).astype(
            {
                'id':int,
                'Cartn_x': float,
                'Cartn_y': float,
                'Cartn_z': float,
                'auth_seq_id': int,
                'pdbx_PDB_model_num': int
            }
        )
        # tab = tab.apply(pd.to_numeric).fillna('')


    elif fmt == 'CIF':
        text = ''
        if os.path.isfile(path):
            with open(path, 'r') as file:
                text = file.read()
        else:
            if len(name) == 4:
                import requests
                url = URL.format(name.upper(), '.cif')
                resp = requests.get(url)
                if resp.status_code == 200:
                    text = resp.text
                else:
                    raise Exception(
                        'Could not obtain {name} structure in {fmt} format from RCSB PDB'.format(
                            name=name, fmt=fmt
                        )
                    )
        
        start = text.find('loop_\n_atom_site.') + len('loop_\n')
        if start == -1 + len('loop_\n'):
            raise Exception(
                'File {} does not contain {} format data'.format(path, fmt)
            )
        
        end   = text.find('#', start) - 1
        tab   = text[start:end].split('\n')
        
        
        columns = []
        for i, line in enumerate(tab):
            if line.startswith('_'):
                columns.append(line.split('.', 1)[1].strip())
            else:
                break
        
        items = map(str.split, tab[i:])
        tab   = pd.DataFrame(items, columns=columns).astype(
            {
                'id':int,
                'Cartn_x': float,
                'Cartn_y': float,
                'Cartn_z': float,
                'auth_seq_id': int,
                'pdbx_PDB_model_num': int
            }
        )
        # tab   = tab.apply(pd.to_numeric)
        
        auth  = [
            'auth_asym_id',
            'auth_seq_id',
            'auth_comp_id',
            'auth_atom_id'
        ]
        label = [
            'label_asym_id',
            'label_seq_id',
            'label_comp_id',
            'label_atom_id'
        ]
        for a, l in zip(auth, label):
            if a not in tab.columns:
                tab[a] = tab[l]
        
        
        l = lambda x: x[1:-1] if x.startswith('"') or x.startswith("'") else x
        tab['label_atom_id'] = list(map(l, tab['label_atom_id']))
        tab['auth_atom_id']  = list(map(l, tab['auth_atom_id']))
        
        for pdb_col in pdb_columns:
            if pdb_col not in tab.columns:
                tab[pdb_col] = '?'

    tab['auth_comp_id'] = tab['auth_comp_id'].astype(str)
    tab['auth_asym_id'] = tab['auth_asym_id'].astype(str)
    

    if 'auth_atom_id' in tab.columns:
        tab['auth_atom_id'] = tab['auth_atom_id'].str.replace('*', "'")
        tab['auth_atom_id'] = tab['auth_atom_id'].replace(
            to_replace={
                'O1P': 'OP1',
                'O2P': 'OP2'
            }
        )
    if 'label_atom_id' in tab.columns:
        tab['label_atom_id'] = tab['label_atom_id'].str.replace('*', "'")
        tab['label_atom_id'] = tab['label_atom_id'].replace(
            to_replace={
                'O1P': 'OP1',
                'O2P': 'OP2'
            }
        )


    struct = Structure(name)
    struct.set_tab(tab)
    struct.set_fmt(fmt)
    
    return struct


def atom_id_PDBformat(atom:'pd.Series'):

    type_symbol = atom['type_symbol']
    atom_id     = atom['auth_atom_id']

    if len(atom_id) == 4:
        return atom_id

    elif isinstance(type_symbol, str) and type_symbol and atom_id.startswith(type_symbol):
        suff = atom_id[len(type_symbol):]

        return '{:>2}'.format(type_symbol) + '{:<2}'.format(suff)

    else:
        if len(atom_id) == 1:
            return atom_id+' '*2
        if len(atom_id) == 2:
            return atom_id+' '
        return atom_id


# PDB save formats
ATOM   = '{group_PDB:<6}{id:>5} {auth_atom_id:>4}{label_alt_id:1}{auth_comp_id:>3}{auth_asym_id:>2}{auth_seq_id:>4}{pdbx_PDB_ins_code:1}   {Cartn_x:>8.3f}{Cartn_y:>8.3f}{Cartn_z:>8.3f}{occupancy:>6}{B_iso_or_equiv:>6}          {type_symbol:>2}{pdbx_formal_charge:>2}\n'
TER    = 'TER   {id:>5}      {auth_comp_id:>3}{auth_asym_id:>2}{auth_seq_id:>4}                                                      \n'
MODEL  = 'MODEL     {:>4}                                                                  \n'
REMARK = 'REMARK 250 CHAIN RENAMING {} -> {}'
ENDMDL = 'ENDMDL                                                                          \n'
