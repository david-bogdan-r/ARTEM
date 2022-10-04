import os

import pandas as pd 
import numpy  as np


pd.to_numeric.__defaults__ = 'ignore', None


class Structure:
    count = 0
    
    def __init__(self, name:str = '') -> None:
        if not name:
            Structure.count += 1
            name = 'struct_{}'.format(Structure.count)
        
        self.name = name
        self.tab  = None # pandas.DataFrame
        self.fmt  = None # str
    
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return '<{} Structure>'.format(self)
    
    
    def rename(self, name:str) -> None:
        self.name = name
    
    def set_tab(self, tab:pd.DataFrame) -> None:
        self.tab = tab
    
    def get_tab(self) -> pd.DataFrame:
        return self.tab
    
    
    def set_fmt(self, fmt:str) -> None:
        self.fmt = fmt
    
    def get_fmt(self) -> str:
        return self.fmt
    
    def saveto(self, folder:str, fmt:str = None) -> None:
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
                tab = tab.replace('.', '')
                tab.replace('?', '', inplace=True)
            
            text = ''
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
    
    
    def apply_transform(self, transform):
        cols  = ['Cartn_x', 'Cartn_y', 'Cartn_z']
        tab   = self.tab.copy()
        
        coord = tab[cols].values
        rot, tran = transform
        tab.loc[:, cols] = np.round(np.dot(coord, rot) + tran, 3)
        
        struct = Structure(self.name)
        struct.set_tab(tab)
        struct.set_fmt(self.fmt)
        
        return struct
    
    
    def get_sub_struct(self, res:str):
        '''
        res = [#[modelIdInt]][/[asymIdStr]][:[compIdStr][_seqIdInt1[compIdStr|_seqIdInt2]]
        '''
        
        tab  = self.tab
        data = {'#': None, '/': None, ':': None}
        
        gen = iter(res)
        c   = next(gen, False)
        while c:
            if c in data.keys():
                data[c] = ''
                cc      = next(gen, False)
                while cc and cc not in data.keys():
                    data[c] += cc
                    cc      = next(gen, False)
                else:
                    c = cc
        
        val = data['#']
        if val:
            val     = int(val)
            cur_tab = tab[tab['pdbx_PDB_model_num'].eq(val)]
        elif val is None:
            val     = tab['pdbx_PDB_model_num'][0]
            cur_tab = tab[tab['pdbx_PDB_model_num'].eq(val)]
        else:
            cur_tab = tab
        
        val = data['/']
        if val:
            val     = cur_tab['auth_asym_id'].dtype.type(val)
            cur_tab = cur_tab[cur_tab['auth_asym_id'].eq(val)]
        
        val = data[':']
        if val:
            vals   = val.split('_')
            res_id = ''
            
            if len(vals) == 2:
                res_id, seq_id = vals
                
                if seq_id.isdigit():
                    seq_id  = int(seq_id)
                    cur_tab = cur_tab[cur_tab['auth_seq_id'].eq(seq_id)]
                else:
                    for i, c in enumerate(seq_id):
                        if not c.isdigit():
                            break
                    seq_id_d = int(seq_id[:i])
                    cur_tab  = cur_tab[cur_tab['auth_seq_id'].eq(seq_id_d)]
                    
                    seq_id_s = seq_id[i:]
                    cur_tab  = cur_tab[cur_tab['auth_comp_id'].eq(seq_id_s)]
            
            elif len(vals) == 3:
                res_id, seq_id_1, seq_id_2 = vals
                
                seq_id_1 = int(seq_id_1)
                seq_id_2 = int(seq_id_2)
                cur_tab  = cur_tab[
                    cur_tab['auth_seq_id'].between(seq_id_1, seq_id_2)
                ]
            
            if res_id:
                cur_tab = cur_tab[cur_tab['auth_comp_id'].eq(res_id)]
        
        struct = Structure(self.name)
        struct.set_tab(cur_tab)
        struct.set_fmt(self.fmt)
        
        return struct
    
    
    def drop_duplicates_alt_id(self, keep:str = 'last') -> None:
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
    
    
    def artem_desc(self, res_repr:dict[str:list]):
        tab  = self.tab.set_index('auth_atom_id')
        mask = tab['pdbx_PDB_model_num'].astype(str)\
               + '.' + tab['auth_asym_id'].astype(str)\
               + '.' + tab['auth_comp_id'].astype(str)\
               + '.' + tab['auth_seq_id'].astype(str)\
               + '.' + tab['pdbx_PDB_ins_code'].astype(str).replace('?', '')
        
        rres = []
        ures = []
        for code, t in tab[['Cartn_x', 'Cartn_y', 'Cartn_z']].groupby(mask, sort=False):
            res_id = code.split('.', 3)[-2]
            if res_id not in res_repr.keys():
                ures.append(code)
                continue
            
            flg = False
            c   = []
            
            for r in res_repr[res_id]:
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



def parser(path:str, fmt:str = 'PDB', name:str = '') -> Structure:
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


ATOM_FORMAT  = '{group_PDB:<6}{id:>5} {auth_atom_id:<4}{label_alt_id:1}{auth_comp_id:>3}{auth_asym_id:>2}{auth_seq_id:>4}{pdbx_PDB_ins_code:1}   {Cartn_x:>8.3f}{Cartn_y:>8.3f}{Cartn_z:>8.3f}{occupancy:>6.2f}{B_iso_or_equiv:>6.2f}          {type_symbol:>2}{pdbx_formal_charge:>2}\n'
TER_FORMAT   = 'TER   {id:>5}      {auth_comp_id:>3}{auth_asym_id:>2}{auth_seq_id:>4}                                                      \n'
MODEL_FORMAT = 'MODEL     {:>4}                                                                  \n'
ENDMDL       = 'ENDMDL                                                                          \n'