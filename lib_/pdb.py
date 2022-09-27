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
        self.tab  = pd.DataFrame()
        self.fmt  = ''
    
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return '<{} Structure>'.format(self)
    
    
    def set_tab(self, tab:pd.DataFrame) -> None:
        self.tab = tab
    
    def get_tab(self) -> pd.DataFrame:
        return self.tab
    
    
    def set_fmt(self, fmt:str) -> None:
        self.fmt = fmt
    
    def get_fmt(self) -> str:
        return self.fmt
    
    
    def get_sub_struct(self, res:str):
        '''
        res = [#[modelIdInt]]\
              [/[asymIdStr]]\
              [:[compIdStr][_seqIdInt1[compIdStr|_seqIdInt2]]
        '''
        
        tab = self.tab
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
            vals    = val.split('_')
            comp_id = ''
            
            if len(vals) == 2:
                comp_id, seq_id = vals
                
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
                comp_id, seq_id_1, seq_id_2 = vals
                
                seq_id_1 = int(seq_id_1)
                seq_id_2 = int(seq_id_2)
                cur_tab  = cur_tab[
                    cur_tab['auth_seq_id'].between(seq_id_1, seq_id_2)
                ]
            
            if comp_id:
                cur_tab = cur_tab[cur_tab['auth_comp_id'].eq(comp_id)]
        
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
    
    
    def artem_desc(self, desc:dict) -> list:
        tab  = self.tab.set_index('auth_atom_id')
        mask = tab['pdbx_PDB_model_num'].astype(str)\
                + '.' + tab['auth_asym_id'].astype(str)\
                + '.' + tab['auth_comp_id'].astype(str)\
                + '.' + tab['auth_seq_id'].astype(str)\
                + '.' + tab['pdbx_PDB_ins_code'].astype(str).replace('?', '')
        
        comp   = []
        # comp_b = []
        for code, t in tab[['Cartn_x', 'Cartn_y', 'Cartn_z']].groupby(mask):
            flg = False
            c   = []
            comp_id = code.split('.', 3)[-2]
            
            if comp_id not in desc:
                # comp_b.append(code)
                continue
            
            comp_desc = desc[comp_id]
            for d in comp_desc:
                m = []
                for dd in d:
                    v = t.loc[t.index.intersection(dd)].values
                    if not len(v):
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
                # comp_b.append(code)
                continue
            
            c[1] = c[1].mean(axis=0)
            comp.append([code, *c])
        
        comp    = list(zip(*comp))
        comp[2] = np.vstack(comp[2])
        return comp #, comp_b


class Parser:
    def __call__(self, path:str, fmt:str = 'PDB', name:str = ''):
        if fmt == 'PDB':
            return Parser.as_pdb(path, name)
        
        if fmt == 'CIF':
            return Parser.as_cif(path, name)
    
    
    @classmethod
    def as_cif(cls, path:str, name:str) -> Structure:
        file = open(path, 'r')
        text = file.read()
        file.close()
        
        start = text.find('_atom_site.')
        end   = text.find('#', start) - 1
        tab   = text[start: end].split('\n')
        
        columns = []
        for i, line in enumerate(tab):
            if line.startswith('_atom_site'):
                columns.append(line.split('.', 1)[1].strip())
            else:
                break
        
        items = map(str.split, tab[i:])
        tab   = pd.DataFrame(items, columns=columns)
        tab   = tab.apply(pd.to_numeric)
        
        l = lambda x: x[1:-1] if x.startswith('"') or x.startswith("'") else x
        
        tab['label_atom_id'] = list(map(l, tab['label_atom_id']))
        if 'auth_atom_id' in tab.columns:
            tab['auth_atom_id'] = list(map(l, tab['auth_atom_id']))
        
        
        struct = Structure(name)
        struct.set_tab(tab)
        struct.set_fmt('CIF')
        
        return struct
    
    
    @classmethod
    def as_pdb(cls, path:str, name:str) -> Structure:
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
        tab['pdbx_PDB_ins_code'].fillna('?', inplace=True)
        
        struct = Structure(name)
        struct.set_tab(tab)
        struct.set_fmt('PDB')
        
        return struct


# tab['label_alt_id'].fillna('.', inplace=True)
# tab['pdbx_PDB_ins_code'].fillna('?', inplace=True)
# tab['pdbx_formal_charge'].fillna('?', inplace=True)

# tab.drop_duplicates(
        #     [
        #         'pdbx_PDB_model_num', 
        #         'auth_asym_id', 
        #         'auth_comp_id',
        #         'auth_seq_id',
        #         'pdbx_PDB_ins_code',
        #         'auth_atom_id'
        #     ],
        #     keep = 'last',
        #     inplace = True
        # )