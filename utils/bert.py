import copy
import math
import json
from attrdict import AttrDict
import collections

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import random
import itertools

import subprocess
import matplotlib.pyplot as plt
import time

#sklearn
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans as KM
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import sys
sys.path.append("/home/aca10223gf/workplace/tools/ViennaRNA/lib/python3.6/site-packages/")
import RNA

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

def get_config(file_path):
    config_file = file_path  # "./weights/bert_config.json"
    json_file = open(config_file, 'r')
    json_object = json.load(json_file)
    config = AttrDict(json_object)
    return config

class visualize_attention():
    def __init__(self, y, swap_kmer_dict):
        self.y = y
        self.swap_kmer_dict = swap_kmer_dict
    
    def highlight(self, word, attn):

        html_color = '#%02X%02X%02X' % (
            255, int(255*(1 - attn)), int(255*(1 - attn)))
        return '<span style="background-color: {}"> {}</span>'.format(html_color, word)


    def mk_html(self, index, sentence, label, normlized_weights):
    #     normlized_weightsは[データindex, multiheadattention, 各単語, 各単語とのattentionの重み]
        label = self.y[int(label)]
        html = str(label) + '<br>'
        
        # # for i in range(12):
        # i = 0
        # attens = normlized_weights[index, i, :, :]
        # attens = attens.sum(0)
        # attens /= attens.max()

        # html += '[BERTのAttentionを可視化_' + str(i+1) + ']<br>'
        # for word, attn in zip(sentence, attens):

        #     if word == 0:
        #         break
        #     html += self.highlight(self.swap_kmer_dict[int(word)], attn)
            
        # html += "<br><br>"

        
        # for i in range(len(sentence)):
        #     attens = normlized_weights[index, :, i, :]
        #     attens = attens.sum(0)
        #     attens /= attens.max()

        #     html += '[BERTのAttentionを可視化_単語' + str(i+1) + ']<br>'
        #     for word, attn in zip(sentence, attens):
        #         if word == 0:
        #             break

        #         html += self.highlight(self.swap_kmer_dict[int(word)], attn)
                
        #     html += "<br><br>"        
            
        # all_attens = attens*0  # all_attensという変数を作成する
        all_attens = normlized_weights[index, :, :, :]
        all_attens = torch.where(all_attens  < 0.2, all_attens * 0, all_attens)

        all_attens = all_attens.sum((0,1))
        all_attens /= all_attens.max()    

        # html += '[BERTのAttentionを可視化_ALL]<br>'
        for word, attn in zip(sentence, all_attens):
            # 単語が[SEP]の場合は文章が終わりなのでbreak
            if word == 0:
                break
            html += self.highlight(self.swap_kmer_dict[int(word)], attn)
        html += "<br>"
        
        return html


def show_base_PCA(features, targets, SS, substructure):
    # from matplotlib import pyplot as plt
    number = 4000
    # 1:"MASK"
    basedict = {0:"PAD", 2:"A", 3:"U", 4:"G", 5:"C"}
    ssdict_pred = {"s":"stem pair", "i":"interior loop", "h":"hairpin loop", "m":"multiloop", "f":"external loop(5)", "t":"external loop(3)"}
    ssdict_ref = {0:"unknown or pad", 1: "external loop", 2:"basepairs in simple stem loops", 3:"basepairs enclosing multifurcations", 4:"pseudoknot", 5:"Bulges and interior loops", 6:"Hairpin loops", 7:"Multibranch loops"}
    transformed = TSNE(n_components=2, perplexity=30.0).fit_transform(features[0:number])
    targets = targets[0:number]
    for label in np.unique(targets)[1:]:
        plt.scatter(transformed[targets == label, 0],
                    transformed[targets == label, 1], label=basedict[label])
    plt.title('tsne')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
    plt.subplots_adjust(right=0.7)
    plt.show()
    plt.savefig("../png/result_{}_base.png".format(int(time.time())))
    plt.close()
    
    substructure = SS.tolist()
    substructure = substructure[0:number]
    for label in list(set(substructure)):
        if not label == 0:
            index = [i for i, x in enumerate(substructure) if x == label]
            plt.scatter(transformed[index, 0],
                        transformed[index, 1], label=ssdict_ref[label])
    plt.title('tsne')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=6)
    plt.subplots_adjust(right=0.7)
    plt.show()
    plt.savefig("../png/result_{}_ss_ALL.png".format(int(time.time())))
    plt.close()

    for label1 in np.unique(targets)[1:]:
        transformed = TSNE(n_components=2).fit_transform(features[0:number][targets == label1])
        for label2 in list(set(substructure)):
            if not label2 == 0:
                index = [i for i, x in enumerate(list(np.array(substructure)[targets == label1])) if x == label2]
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], label=ssdict_ref[label2])
        plt.title(basedict[label1])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=6)
        plt.subplots_adjust(right=0.7)
        plt.show()
        plt.savefig("../png/result_{}_ss_{}.png".format(int(time.time()), basedict[label1]))
        plt.close()



class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weightのこと
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # biasのこと
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        words_embeddings = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()

        self.attention = BertAttention(config)

        self.intermediate = BertIntermediate(config)

        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        if attention_show_flg == True:
            attention_output, attention_probs = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs

        elif attention_show_flg == False:
            attention_output = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output  # [batch_size, seq_length, hidden_size]


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.selfattn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        if attention_show_flg == True:
            self_output, attention_probs = self.selfattn(
                input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output, attention_probs

        elif attention_show_flg == False:
            self_output = self.selfattn(
                input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        # num_attention_heads': 12

        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads) 
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
 
        if attention_show_flg == True:
            return context_layer, attention_probs
        elif attention_show_flg == False:
            return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])
        # self.layer = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size)
        #                             for _ in range(config.num_hidden_layers)])                            

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, attention_show_flg=False):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            if attention_show_flg == True:
                hidden_states, attention_probs = layer_module(
                    hidden_states, attention_mask, attention_show_flg)
            elif attention_show_flg == False:
                hidden_states = layer_module(
                    hidden_states, attention_mask, attention_show_flg)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        if attention_show_flg == True:
            return all_encoder_layers, attention_probs
        elif attention_show_flg == False:
            return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense(first_token_tensor)

        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, attention_show_flg=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        if attention_show_flg == True:
            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers, attention_show_flg)

        elif attention_show_flg == False:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers, attention_show_flg)

        pooled_output = self.pooler(encoded_layers[-1])

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if attention_show_flg == True:
            return encoded_layers, pooled_output, attention_probs
        elif attention_show_flg == False:
            return encoded_layers, pooled_output


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config ):
        super(BertPreTrainingHeads, self).__init__()

        self.predictions = MaskedWordPredictions(config)
        config.vocab_size = config.ss_size
        self.predictions_ss = MaskedWordPredictions(config)

        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        prediction_scores_ss = self.predictions_ss(sequence_output)

        seq_relationship_score = self.seq_relationship(
            pooled_output)

        return prediction_scores, prediction_scores_ss, seq_relationship_score


class MaskedWordPredictions(nn.Module):
    def __init__(self, config):
        super(MaskedWordPredictions, self).__init__()

        self.transform = BertPredictionHeadTransform(config)
        

        self.decoder = nn.Linear(in_features=config.hidden_size, 
                                 out_features=config.vocab_size,
                                 bias=False)
        self.bias = nn.Parameter(torch.zeros(
            config.vocab_size)) 

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.transform_act_fn = gelu

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states



class SeqRelationship(nn.Module):
    def __init__(self, config, out_features):
        super(SeqRelationship, self).__init__()

        self.seq_relationship = nn.Linear(config.hidden_size, out_features)

    def forward(self, pooled_output):
        return self.seq_relationship(pooled_output)


class BertForMaskedLM(nn.Module):
    def __init__(self, config, net_bert):
        super(BertForMaskedLM, self).__init__()

        self.bert = net_bert 

        self.cls = BertPreTrainingHeads(config)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, attention_show_flg=False):
        if attention_show_flg == False:
            encoded_layers, pooled_output = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=False)
            
        else:
            encoded_layers, pooled_output, attention_probs = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=True)

        prediction_scores, prediction_scores_ss, seq_relationship_score = self.cls(encoded_layers, pooled_output)
        return prediction_scores, prediction_scores_ss, encoded_layers


def set_learned_params(net, weights_path):
    loaded_state_dict = torch.load(weights_path)
    net.eval()
    param_names = []
    for name, param in net.named_parameters():
        param_names.append(name)
    new_state_dict = net.state_dict().copy()
    for index, (key_name, value) in enumerate(loaded_state_dict.items()):
        name = param_names[index]
        new_state_dict[name] = value 
        # print(str(key_name)+"→"+str(name))
        if (index+1 - len(param_names)) >= 0:
            break
    net.load_state_dict(new_state_dict)
    return net


def fix_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.bert.encoder.layer[-1].named_parameters():
        param.requires_grad = True
    for name, param in model.bert.encoder.layer[-2].named_parameters():
        param.requires_grad = True
    for name, param in model.cls.named_parameters():
        param.requires_grad = True
    for name, param in model.bert.embeddings.named_parameters():
        param.requires_grad = False
    return model