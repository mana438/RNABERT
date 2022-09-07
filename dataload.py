import numpy as np
import random
from Bio import SeqIO
import torch 
from torchvision import transforms, datasets
import copy
import itertools
from Bio import AlignIO
from Bio.Alphabet import generic_rna

class DATA:
    def __init__(self, args, config):
        self.max_length = config.max_position_embeddings
        self.mag = args.mag
        self.maskrate = args.maskrate 
        self.batch_size = args.batch

    def load_data_MLM_SFP(self, data_sets):
        families = []
        gapped_seqs = []
        seqs = []
        for i, data_set in enumerate(data_sets):
            for record in SeqIO.parse(data_set, "fasta"):
                gapped_seq = str(record.seq).upper()
                gapped_seq = gapped_seq.replace("T", "U")
                seq = gapped_seq.replace('-', '')
                if set(seq) <= set(['A', 'T', 'G', 'C', 'U']) and len(list(seq)) < self.max_length:
                    seqs.append(seq)
                    families.append(i)
                    gapped_seqs.append(gapped_seq)
        gapped_seqs = np.tile(onehot_seq(gapped_seqs, self.max_length*5), (self.mag, 1))
        family = np.tile(np.array(families), self.mag)
        seqs_len = np.tile(np.array([len(i) for i in seqs]), self.mag)   
        k = 1   
        kmer_seqs = kmer(seqs, k)
        masked_seq, low_seq = mask(kmer_seqs, rate = self.maskrate, mag = self.mag)
        kmer_dict = make_dict(k)
        swap_kmer_dict = {v: k for k, v in kmer_dict.items()}
        masked_seq = np.array(convert(masked_seq, kmer_dict, self.max_length))
        low_seq = np.array(convert(low_seq, kmer_dict, self.max_length))

        transform = transforms.Compose([transforms.ToTensor()])
        low_seq_1, masked_seq_1, family_1, seqs_len_1 = self.sfp_data(low_seq, masked_seq, family, seqs_len, 0.5)
        ds_MLM_SFP = MyDataset("MLM", low_seq, masked_seq, family, seqs_len, low_seq_1, masked_seq_1, family_1, seqs_len_1)
        dl_MLM_SFP = torch.utils.data.DataLoader(ds_MLM_SFP, self.batch_size, shuffle=True)
        return dl_MLM_SFP

    def load_data_EMB(self, data_sets):
        families = []
        gapped_seqs = []
        seqs = []
        for i, data_set in enumerate(data_sets):
            for record in SeqIO.parse(data_set, "fasta"):
                gapped_seq = str(record.seq).upper()
                gapped_seq = gapped_seq.replace("T", "U")
                seq = gapped_seq.replace('-', '')
                if set(seq) <= set(['A', 'T', 'G', 'C', 'U']) and len(list(seq)) < self.max_length:
                    seqs.append(seq)
                    families.append(i)
                    gapped_seqs.append(gapped_seq)
        gapped_seqs = np.tile(onehot_seq(gapped_seqs, self.max_length*5), (self.mag, 1))
        family = np.tile(np.array(families), self.mag)
        seqs_len = np.tile(np.array([len(i) for i in seqs]), self.mag)   
        k = 1   
        kmer_seqs = kmer(seqs, k)
        masked_seq, low_seq = mask(kmer_seqs, rate = 0, mag = self.mag)
        kmer_dict = make_dict(k)
        swap_kmer_dict = {v: k for k, v in kmer_dict.items()}
        masked_seq = np.array(convert(masked_seq, kmer_dict, self.max_length))
        low_seq = np.array(convert(low_seq, kmer_dict, self.max_length))

        transform = transforms.Compose([transforms.ToTensor()])
        ds_MLM_SFP_ALIGN = MyDataset("SHOW", low_seq, masked_seq, family, seqs_len)
        dl_MLM_SFP_ALIGN = torch.utils.data.DataLoader(ds_MLM_SFP_ALIGN, self.batch_size, shuffle=False)
        return seqs, low_seq, dl_MLM_SFP_ALIGN 

    def load_data_MUL(self, data_sets, train_type):
        families = []
        gapped_seqs = []
        seqs = []
        seqs_len = []
        for i, data_set in enumerate(data_sets):
            num = len(list(SeqIO.parse(data_set, "fasta")))
            num = num if num < self.mag else self.mag
            for j, record in enumerate(SeqIO.parse(data_set, "fasta")):
                gapped_seq = str(record.seq).upper().replace("T", "U")
                seq = gapped_seq.replace('-', '').replace('.', '')
                if set(seq) <= set(['A', 'T', 'G', 'C', 'U']) and len(list(seq)) < self.max_length:
                    seqs.extend([seq] * num)
                    seqs_len.extend([len(seq)] * num)
                    families.extend([i] * num)
                    gapped_seqs.extend([gapped_seq] * num)
                if train_type == "ALN" and j == 0:
                    break
        gapped_seqs = onehot_seq(gapped_seqs, self.max_length*5)
        family = np.array(families)
        seqs_len = np.array(seqs_len)
        # PAD 0, mask 1, A 2, U 3, G 4 ,C 5, 
        low_seq = base_to_num(seqs, self.max_length)
        masked_seq = mask_seq(low_seq, rate = self.maskrate)
        low_seq_1, masked_seq_1, family_1, seqs_len_1, common_index, common_index_1 = self.sfp_data(low_seq, masked_seq, family, seqs_len, family_ratio = 0.0, gapped_seqs = gapped_seqs)
        ds_MLM_SFP_ALIGN = MyDataset("MUL", low_seq, masked_seq, family, seqs_len, low_seq_1, masked_seq_1, family_1, seqs_len_1, common_index, common_index_1)
        dl_MLM_SFP_ALIGN = torch.utils.data.DataLoader(ds_MLM_SFP_ALIGN, self.batch_size, shuffle=True)
        return dl_MLM_SFP_ALIGN

    def load_data_SSL(self, data_sets):
        families = []
        gapped_seqs = []
        seqs = []
        SS = []
        for i, data_set in enumerate(data_sets):
            align = AlignIO.read(data_set, "stockholm", alphabet=generic_rna)
            cons_SS = align.column_annotations["secondary_structure"]
            for j, record in enumerate(align):
                gapped_seq = str(record.seq).upper()
                gapped_seq = gapped_seq.replace("T", "U")
                seq = gapped_seq.replace('-', '').replace('.', '')
                ss = ''.join([cons_SS[i] for i, s in enumerate( list(gapped_seq)) if s != "-"])
                if set(seq) <= set(['A', 'T', 'G', 'C', 'U']) and len(list(seq)) < self.max_length:
                    seqs.append(seq)
                    families.append(i)
                    gapped_seqs.append(gapped_seq)
                    SS.append(ss)
        gapped_seqs = np.tile(onehot_seq(gapped_seqs, self.max_length*5), (self.mag, 1))
        SS = np.tile(secondary_num(SS, self.max_length), (self.mag, 1))
        family = np.tile(np.array(families), self.mag)
        seqs_len = np.tile(np.array([len(i) for i in seqs]), self.mag)
        k = 1   
        kmer_seqs = kmer(seqs, k)
        # PAD 0, mask 1, A 2, U 3, G 4 ,C 5, 
        masked_seq, low_seq = mask(kmer_seqs, rate = self.maskrate, mag = self.mag)
        kmer_dict = make_dict(k)
        swap_kmer_dict = {v: k for k, v in kmer_dict.items()}
        masked_seq = np.array(convert(masked_seq, kmer_dict, self.max_length))
        low_seq = np.array(convert(low_seq, kmer_dict, self.max_length))

        transform = transforms.Compose([transforms.ToTensor()])
        low_seq_1, masked_seq_1, family_1, seqs_len_1, common_index, common_index_1, SS_1 = self.sfp_data(low_seq, masked_seq, family, seqs_len, family_ratio = 0.0, gapped_seqs = gapped_seqs, SS = SS)
        ds_MLM_SFP_ALIGN = MyDataset("SSL", low_seq, masked_seq, family, seqs_len, low_seq_1, masked_seq_1, family_1, seqs_len_1, common_index, common_index_1, SS, SS_1)
        dl_MLM_SFP_ALIGN = torch.utils.data.DataLoader(ds_MLM_SFP_ALIGN, self.batch_size, shuffle=True)
        return dl_MLM_SFP_ALIGN

    def load_data_SHOW(self, data_sets):
        import forgi.graph.bulge_graph as fgb
        families = []
        gapped_seqs = []
        seqs = []
        SS = []
        for i, data_set in enumerate(data_sets):
            align = AlignIO.read(data_set, "stockholm", alphabet=generic_rna)
            cons_SS = align.column_annotations["secondary_structure"]
            for j, record in enumerate(align):
                gapped_seq = str(record.seq).upper()
                gapped_seq = gapped_seq.replace("T", "U")
                seq = gapped_seq.replace('-', '')
                ss = ''.join([cons_SS[i] for i, s in enumerate( list(gapped_seq)) if s != "-"])
                if set(seq) <= set(['A', 'T', 'G', 'C', 'U']) and len(list(seq)) < self.max_length:
                    try:
                        fgb.BulgeGraph.from_dotbracket(ss)
                    except:
                        print('Too many closing brackets') 
                    else:
                        seqs.append(seq)
                        families.append(i)
                        gapped_seqs.append(gapped_seq)
                        SS.append(ss)
        gapped_seqs = np.tile(onehot_seq(gapped_seqs, self.max_length*5), (self.mag, 1))
        # SS = np.tile(secondary_num(SS, self.max_length), (self.mag, 1))
        SS = list(itertools.chain.from_iterable([list(fgb.BulgeGraph.from_dotbracket(db).to_element_string().ljust(440, 'X')) for db in SS]))
        family = np.tile(np.array(families), self.mag)
        seqs_len = np.tile(np.array([len(i) for i in seqs]), self.mag)
        k = 1   
        kmer_seqs = kmer(seqs, k)

        masked_seq, low_seq = mask(kmer_seqs, rate = self.maskrate, mag = self.mag)
        kmer_dict = make_dict(k)
        swap_kmer_dict = {v: k for k, v in kmer_dict.items()}
        masked_seq = np.array(convert(masked_seq, kmer_dict, self.max_length))
        low_seq = np.array(convert(low_seq, kmer_dict, self.max_length))

        transform = transforms.Compose([transforms.ToTensor()])
        ds_MLM_SFP_ALIGN = MyDataset("SHOW", low_seq, masked_seq, family, seqs_len)
        dl_MLM_SFP_ALIGN = torch.utils.data.DataLoader(ds_MLM_SFP_ALIGN, self.batch_size, shuffle=False)
        return seqs, low_seq, SS, ds_MLM_SFP_ALIGN, dl_MLM_SFP_ALIGN 


    def load_data_CLU(self, data_sets):
        families = []
        gapped_seqs = []
        seqs = []
        for i, data_set in enumerate(data_sets):
            for record in SeqIO.parse(data_set, "fasta"):
                gapped_seq = str(record.seq).upper()
                gapped_seq = gapped_seq.replace("T", "U")
                seq = gapped_seq.replace('-', '')
                if set(seq) <= set(['A', 'T', 'G', 'C', 'U']) and len(list(seq)) < self.max_length:
                    seqs.append(seq)
                    families.append(i)
                    gapped_seqs.append(gapped_seq)
        gapped_seqs = np.tile(onehot_seq(gapped_seqs, self.max_length*5), (self.mag, 1))
        family = np.tile(np.array(families), self.mag)
        seqs_len = np.tile(np.array([len(i) for i in seqs]), self.mag)   
        k = 1   
        kmer_seqs = kmer(seqs, k)
        masked_seq, low_seq = mask(kmer_seqs, rate = 0, mag = 1)
        kmer_dict = make_dict(k)
        swap_kmer_dict = {v: k for k, v in kmer_dict.items()}
        # masked_seq = np.array(convert(masked_seq, kmer_dict, self.max_length))
        low_seq = np.array(convert(low_seq, kmer_dict, self.max_length))
        transform = transforms.Compose([transforms.ToTensor()])
        ds_CLU = MyDataset("CLU", low_seq, masked_seq, family, seqs_len)
        dl_CLU = torch.utils.data.DataLoader(ds_CLU, self.batch_size, shuffle=False)
        return seqs, low_seq, ds_CLU, dl_CLU 

    def sfp_data(self, low_seq, masked_seq, family, seqs_len, family_ratio = 0.5, gapped_seqs = None, SS = None):
        new_index = []
        for i in range(len(family)):
            if random.random() >= family_ratio:
                indices = np.where(family == family[i])[0]
            else:
                indices = np.where(family != family[i])[0]
            # eliminate himself 
            indices = np.delete(indices, np.where(indices == i)[0])
            if indices.size != 0:
                index = np.random.choice(indices, 1, replace=False)
                new_index.append(index[0])
            else:
                new_index.append(i)
        new_seqs_len = seqs_len[new_index]
        new_family = family[new_index]
        if gapped_seqs is not None:
            new_gapped_seqs = gapped_seqs[new_index]
            A1 = gapped_seqs + gapped_seqs * new_gapped_seqs
            B1 = new_gapped_seqs + gapped_seqs * new_gapped_seqs
            A1 = [i[np.where(i != 0)]-1 for i in A1]
            A1 = np.array([np.pad(i, ((0, self.max_length-len(i)))) for i in A1])
            B1 = [i[np.where(i != 0)]-1 for i in B1]
            B1 = np.array([np.pad(i, ((0, self.max_length-len(i)))) for i in B1])
        new_low_seq = low_seq[new_index, :]
        new_masked_seq = masked_seq[new_index, :]
        if gapped_seqs is not None and SS is not None:
            new_SS = SS[new_index, :]
            return new_low_seq, new_masked_seq, new_family, new_seqs_len, A1, B1, new_SS
        elif gapped_seqs is not None:
            return new_low_seq, new_masked_seq, new_family, new_seqs_len, A1, B1
        else:
            return new_low_seq, new_masked_seq, new_family, new_seqs_len

    def split_dataset(self, ds, train_size, ds1=None):
        n_samples = len(ds)
        subset_indices = random.sample(range(n_samples), k=train_size)
        # separate after shuffle
        train_dataset = Subset(ds, subset_indices)
        if ds1:
            train_dataset1 = Subset(ds1, subset_indices)
            return train_dataset, train_dataset1
        else:
            return train_dataset


def base_to_num(seq, pad_max_length):
    seq = [list(i.translate(str.maketrans({'A': "2", 'U': "3", 'G': "4", 'C': "5"}))) for i in seq]
    seq = [list(map(lambda x : int(x), s)) for s in seq]
    seq = np.array([np.pad(s, ((0, pad_max_length-len(s)))) for s in seq])
    return seq

def num_to_base(seq):
    seq = seq.tolist()
    seq = ["".join(map(str, i)).replace('0', '').translate(str.maketrans({'2': "A", '3': "U", '4': "G", '5': "C"})) for i in seq]
    return seq


def mask_seq(seqs, rate = 0.2):
    c = np.random.rand(*seqs.shape)
    masked_seqs = np.where((c < 0.2) & (seqs != 0) , 1, seqs)
    d = np.random.randint(2, 6, c.shape)
    masked_seqs = np.where((c < 0.02) & (seqs != 0) , d, masked_seqs)
    return masked_seqs


def onehot_seq(gapped_seq, pad_max_length):
    gapped_seq = [list(i.translate(str.maketrans({'-': "0", '.' : "0", 'A': "1", 'U': "1", 'G': "1", 'C': "1"}))) for i in gapped_seq]
    gapped_seq = [list(map(lambda x : int(x), s)) for s in gapped_seq]
    gapped_seq = np.array([np.pad(s, ((0, pad_max_length-len(s)))) for s in gapped_seq])
    return gapped_seq

def secondary_num(SS, pad_max_length):
    SS = [list(i.translate(str.maketrans({'.': "0", ':': "1", '<': "2", '>': "2", '(': "3", ')': "3", '{': "3", '}': "3", '[': "3", ']': "3", 'A': "4", 'a': "4", 'B': "4", 'b': "4", '-': "5", '_': "6", ',': "7"}))) for i in SS]
    SS = [list(map(lambda x : int(x), s)) for s in SS]
    SS = np.array([np.pad(s, ((0, pad_max_length-len(s)))) for s in SS])
    return SS

def kmer(seqs, k=1):
    #塩基文字列をk-mer文字列リストに変換
    kmer_seqs = []
    for seq in seqs:
        kmer_seq = []
        for i in range(len(seq)):
            if i <= len(seq)-k:
                kmer_seq.append(seq[i:i+k])
        kmer_seqs.append(kmer_seq)
    return kmer_seqs
            
def mask(seqs, rate = 0.2, mag = 1):
    # 与えられた文字列リストに対してmask。rateはmaskの割合,magは生成回数/1配列
    seq = []
    masked_seq = []
    label = []
    for i in range(mag):
        seqs2 = copy.deepcopy(seqs)
        for s in seqs2:
            label.append(copy.copy(s))
            mask_num = int(len(s)*rate)
            all_change_index = np.array(random.sample(range(len(s)), mask_num))
            mask_index, base_change_index = np.split(all_change_index, [int(all_change_index.size * 0.90)])
#             index = list(np.sort(random.sample(range(len(s)), mask_num)))
            for i in list(mask_index):
                s[i] = "MASK"
            for i in list(base_change_index):
                s[i] = random.sample(('A', 'U', 'G', 'C'), 1)[0] 
            masked_seq.append(s)
    return masked_seq, label

def seq_label(seqs):
    return seqs

def convert(seqs, kmer_dict, max_length):
    # 文字列リストを数字に変換
    seq_num = [] 
    if not max_length:
        max_length = max([len(i) for i in seqs])
    for s in seqs:
        convered_seq = [kmer_dict[i] for i in s] + [0]*(max_length - len(s))
        seq_num.append(convered_seq)
    return seq_num

def make_dict(k=3):
    # seq to num 
    l = ["A", "U", "G", "C"]
    kmer_list = [''.join(v) for v in list(itertools.product(l, repeat=k))]
    kmer_list.insert(0, "MASK")
    dic = {kmer: i+1 for i,kmer in enumerate(kmer_list)}
    return dic


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train_type, low_seq, masked_seq, family, seq_len, low_seq_1 = None, masked_seq_1 = None, family_1 = None, seq_len_1 = None, common_index = None, common_index_1 = None, SS = None, SS_1 = None):
        self.train_type = train_type
        self.data_num = len(low_seq)
        self.low_seq = low_seq
        self.low_seq_1 = low_seq_1
        self.masked_seq = masked_seq
        self.masked_seq_1 = masked_seq_1
        self.family = family
        self.family_1 = family_1
        self.seq_len = seq_len
        self.seq_len_1 = seq_len_1
        self.common_index = common_index 
        self.common_index_1 = common_index_1 
        self.SS = SS
        self.SS_1 = SS_1
    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_low_seq = self.low_seq[idx]
        out_masked_seq = self.masked_seq[idx]
        out_family = self.family[idx]
        out_seq_len = self.seq_len[idx]
        if self.train_type == "MLM" or self.train_type == "MUL" or self.train_type == "SSL":
            out_low_seq_1 = self.low_seq_1[idx]
            out_masked_seq_1 = self.masked_seq_1[idx]
            out_family_1 = self.family_1[idx]
            out_seq_len_1 = self.seq_len_1[idx]

        if self.train_type == "MUL" or self.train_type == "SSL":
            out_common_index = self.common_index[idx]
            out_common_index_1 = self.common_index_1[idx]

        if self.train_type == "SSL":
            out_SS = self.SS
            out_SS_1 = self.SS_1

        # if self.train_type == "SHOW":
        #     out_SS = self.SS

        if self.train_type == "MLM":
            return out_low_seq, out_masked_seq, out_family, out_seq_len, out_low_seq_1, out_masked_seq_1, out_family_1, out_seq_len_1
        elif self.train_type == "MUL":
            return out_low_seq, out_masked_seq, out_family, out_seq_len, out_low_seq_1, out_masked_seq_1, out_family_1, out_seq_len_1, out_common_index, out_common_index_1
        elif self.train_type == "SSL":
            return out_low_seq, out_masked_seq, out_family, out_seq_len, out_low_seq_1, out_masked_seq_1, out_family_1, out_seq_len_1, out_common_index, out_common_index_1, out_SS, out_SS_1
        # elif self.train_type == "SHOW":
        #     return out_low_seq, out_family, out_seq_len, out_SS
        else:
            return out_low_seq, out_family, out_seq_len
