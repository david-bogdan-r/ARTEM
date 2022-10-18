import os

import pandas as pd 
import numpy  as np


pd.to_numeric.__defaults__ = 'ignore', None


class Structure:
    count = 0
    
    def __init__(self, name:'str' = '') -> 'None':
        if not name:
            Structure.count += 1
            name = 'struct_{}'.format(Structure.count)
        
        self.name = name
        self.tab  = None # pandas.DataFrame
        self.fmt  = None # str
    
    
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
    
    
    def one_letter_chain_renaming(tab:'pd.DataFrame'):
        if not hasattr(Structure, 'chain_labels'):
            from string import ascii_letters, digits
            chain_labels = sorted(ascii_letters) + list(digits)
            Structure.chain_labels = chain_labels
        else:
            chain_labels = Structure.chain_labels
        
        cur_chain_labels = tab['auth_asym_id'].astype(str).unique()
        allowed_labels = sorted(set(chain_labels) - set(cur_chain_labels))
        
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
                rmk = REMARK_FORMAT.format(k, v)
                rmk += ' ' * (80 - len(rmk)) + '\n'
                msg.append(rmk)
        
        return tab, msg

    
    def saveto(self, folder:'str', fmt:'str' = None) -> 'None':
        os.makedirs(folder, exist_ok=True)
        tab  = self.tab
        
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
                tab, msg = Structure.one_letter_chain_renaming(tab)
                tab = tab.replace('.', '')
                tab.replace('?', '', inplace=True)
            else:
                msg = []
            
            text = ''.join(msg)
            for model_num, tt in tab.groupby('pdbx_PDB_model_num', sort=False):
                text += MODEL_FORMAT.format(model_num)
                tt['id'] = range(1, len(tt) + 1)
                chain_count = 0
                for asym_id, ttt in tt.groupby('auth_asym_id', sort=False):
                    ttt['id'] = (ttt['id'] + chain_count) % 1_000_000
                    for item in ttt.iloc:
                        text += ATOM_FORMAT.format(**item)
                    item['id'] += 1
                    text  += TER_FORMAT.format(**item)
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
                tab['label_seq_id']    = tab['auth_seq_id']
                
                entity = tab['label_asym_id'].unique()
                entity = dict(zip(entity, range(1, len(entity) + 1)))
                tab['label_entity_id'] = tab['label_asym_id'].replace(entity)
            
            
            title = 'data_{}\n'.format(self.name.upper())
            file.write(title)
            
            header  = '# \nloop_\n'
            for col in tab.columns:
                header += '_atom_site.{}\n'.format(col)
            file.write(header)
            tab.to_string(file, header=False, index=False)
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
        '''
        res = [#[modelIdInt]][/[asymIdStr]][:[compIdStr][_seqIdInt1[compIdStr|_seqIdInt2]]
        '''
        
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
        
        if spl['#']:
            spl['#'] = int(spl['#'])
        else:
            if spl['#'] == None:
                spl['#'] = 1
        
        if spl[':']:
            rng = spl[':'].split('_')
            if len(rng) == 3:
                rng[1] = int(rng[1])
                rng[2] = int(rng[2])
            elif len(rng) == 2:
                if rng[1].isdigit():
                    rng[1] = int(rng[1])
            spl[':'] = rng
        
        return spl
    
    
    def get_res_msk(self, res: 'str') -> 'pd.Series':
        tab = self.tab
        spl = Structure._res_split(res)
        
        mod = spl['#']
        if mod:
            mod_msk = tab['pdbx_PDB_model_num'].eq(mod)
        else:
            mod_msk = tab['pdbx_PDB_model_num'].astype(bool)
        
        chn = spl['/']
        if chn:
            chn_msk = tab['auth_asym_id'].eq(chn)
        else:
            chn_msk = tab['auth_asym_id'].astype(bool)
        
        rng  = spl[':']
        if rng:
            case = len(rng)
        else:
            case = 0
        
        # res wo ':'
        if case == 0:
            rng_msk = tab['auth_seq_id'].astype(bool)
       
        # ':N'
        elif case == 1:
            res = rng[0]
            if res:
                rng_msk = tab['auth_comp_id'].eq(rng[0])
            else:
                rng_msk = tab['auth_comp_id'].astype(bool)
        
        #':_num' | ':_numN' | ':N_num' | ':N1_numN2'
        elif case == 2:
            res, num  = rng
            if type(num) == int:
                if res:
                    rng_msk = tab['auth_seq_id'].eq(num) \
                        & tab['auth_comp_id'].eq(res)
                else:
                    rng_msk = tab['auth_seq_id'].eq(num)
            else:
                # ':N1_numN2' = ':_numN2' even if N1 != N2
                dgt = ''
                for i, c in enumerate(num):
                    if c.isdigit():
                        dgt += c
                    else:
                        break
                res = num[i:]
                num = int(dgt)
                rng_msk = tab['auth_seq_id'].eq(num) \
                    & tab['auth_comp_id'].eq(res)
        
        # ':_num1_num2' | ':N_num1_num2'
        elif case == 3:
            res, num_1, num_2 = rng
            if res:
                rng_msk = tab['auth_comp_id'].eq(res) \
                    & tab['auth_seq_id'].between(num_1, num_2)
            else:
                rng_msk = tab['auth_seq_id'].between(num_1, num_2)
        
        # incorrect res
        else:
            rng_msk = tab['auth_seq_id'].astype(bool) ^ True
        
        msk = mod_msk & chn_msk & rng_msk
        
        return msk
    
    
    def get_res_substruct(self, res:'str', neg:'bool' = False) -> 'Structure':
        
        tab = self.tab
        msk = self.get_res_msk(res)
        if neg:
            msk ^=True
        tab = tab[msk]
        
        structure = Structure(self.name)
        structure.set_tab(tab)
        structure.set_fmt(self.fmt)
        
        return structure
    
    def add_code_msk(self) -> None:
        tab = self.tab
        code_msk = tab['pdbx_PDB_model_num'].astype(str)\
            + '.' + tab['auth_asym_id'].astype(str)\
            + '.' + tab['auth_comp_id'].astype(str)\
            + '.' + tab['auth_seq_id'].astype(str)\
            + '.' + tab['pdbx_PDB_ins_code'].astype(str).replace('?', '')
        self.code_msk = code_msk
    
    
    def get_res_code(self, res:'str' = '', neg:'bool' = False) -> 'list':
        if not hasattr(self, 'code_msk'):
            self.add_code_msk()
        
        code_msk = self.code_msk
        msk = self.get_res_msk(res)
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
            self.add_code_msk()
        
        tab  = self.tab.set_index('auth_atom_id')
        code_msk = self.code_msk.set_axis(tab.index)
        
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



def parser(path:'str', fmt:'str' = 'PDB', name:'str' = '') -> 'Structure':
    if fmt == 'PDB':
        columns = (
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
            
            'pdbx_PDB_model_num',
        )
        
        item_mask = (
            slice(0,   6),  # group_PDB
            slice(6,  11),  # id
            slice(12, 16),  # auth_atom_id
            slice(16, 17),  # label_alt_id
            slice(17, 20),  # auth_comp_id
            slice(20, 22),  # auth_asym_id
            slice(22, 26),  # auth_seq_id
            slice(26, 27),  # pdbx_PDB_ins_code
            slice(30, 38),  # Cartn_x
            slice(38, 46),  # Cartn_y
            slice(46, 54),  # Cartn_z
            slice(54, 60),  # occupancy
            slice(60, 66),  # B_iso_or_equiv
            slice(76, 78),  # type_symbol
            slice(78, 80)   # pdbx_formal_charge
        )
        
        rec_names = {'ATOM  ', 'HETATM'}
        rec_name  = slice(0, 6)
        
        items     = []
        cur_model = [1]
        
        file = open(path, 'r')
        for line in file:
            rec = line[rec_name]
            if rec in rec_names:
                item = map(line.__getitem__, item_mask)
                item = map(str.strip, item)
                items.append(list(item) + cur_model)
            elif rec == 'MODEL ':
                cur_model = [int(line.split()[1])]
        file.close()
        
        tab = pd.DataFrame(items, columns=columns)
        tab = tab.apply(pd.to_numeric)
        tab.fillna('', inplace=True)
    
    elif fmt == 'CIF':
        file = open(path, 'r')
        text = file.read()
        file.close()
        
        start = text.find('_atom_site.')
        end   = text.find('#', start) - 1
        tab   = text[start:end].split('\n')
        
        columns = []
        for i, line in enumerate(tab):
            if line.startswith('_'):
                columns.append(line.split('.', 1)[1].strip())
            else:
                break
        
        items = map(str.split, tab[i:])
        tab   = pd.DataFrame(items, columns=columns)
        tab   = tab.apply(pd.to_numeric)
        
        
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
    
    struct = Structure(name)
    struct.set_tab(tab)
    struct.set_fmt(fmt)

    return struct


ATOM_FORMAT   = '{group_PDB:<6}{id:>5} {auth_atom_id:<4}{label_alt_id:1}{auth_comp_id:>3}{auth_asym_id:>2}{auth_seq_id:>4}{pdbx_PDB_ins_code:1}   {Cartn_x:>8.3f}{Cartn_y:>8.3f}{Cartn_z:>8.3f}{occupancy:>6.2f}{B_iso_or_equiv:>6.2f}          {type_symbol:>2}{pdbx_formal_charge:>2}\n'
TER_FORMAT    = 'TER   {id:>5}      {auth_comp_id:>3}{auth_asym_id:>2}{auth_seq_id:>4}                                                      \n'
MODEL_FORMAT  = 'MODEL     {:>4}                                                                  \n'
ENDMDL        = 'ENDMDL                                                                          \n'
REMARK_FORMAT = 'REMARK 250 CHAIN RENAMING {} -> {}'
