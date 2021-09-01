import torch 
import alignment_C as Aln_C
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from dataload import  num_to_base
import time
import numpy as np
class Train_Module:
	def __init__(self, config):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.config = config

	def train_MLM(self, low_seq_0, masked_seq_0, prediction_scores):
		criterion = nn.CrossEntropyLoss()
		mask = masked_seq_0 - low_seq_0 != 0
		length_mask = masked_seq_0 != 0
		same_base_mask = torch.bernoulli(torch.ones(mask.shape)*0.05).byte().to(self.device)  
		mask = mask + same_base_mask
		index = torch.nonzero(mask).split(1, dim=1)
		prediction_scores = torch.squeeze(prediction_scores[index])
		new_low_seq_0 = torch.squeeze(low_seq_0[index])
		loss = criterion(prediction_scores, new_low_seq_0)     
		_, preds = torch.max(prediction_scores, 1)

		correct = torch.sum(preds == new_low_seq_0.data).double()/len(new_low_seq_0)
		return loss, correct

	def train_SSL(self, SS, prediction_scores):
		criterion = nn.CrossEntropyLoss()
		index = torch.nonzero(SS).split(1, dim=1)
		prediction_scores = torch.squeeze(prediction_scores[index])
		SS_answer = torch.squeeze(SS[index])
		
		loss = criterion(prediction_scores, SS_answer)        
		_, preds = torch.max(prediction_scores, 1)
		correct = torch.sum(preds == SS_answer.data).double()/len(SS_answer)
		return loss, correct

	def train_SFP(self, low_seq_0, seq_len_0, low_seq_1, seq_len_1, family_0, family_1, z0_list, z1_list):
		distance_label = torch.where((family_0 - family_1) == 0,  torch.ones(family_0.size()[0]), torch.full((family_0.size()[0],), -1) ).to("cuda")    
		distance_label_same = torch.where((family_0 - family_1) == 0,  torch.ones(family_0.size()[0]), torch.zeros(family_0.size()[0]) ).to("cuda")    
		distance_label_diff = torch.where((family_0 - family_1) != 0,  torch.ones(family_0.size()[0]), torch.zeros(family_0.size()[0]) ).to("cuda")    
		_, logits = self.match(z0_list, z1_list)
		distance = -logits
		distance_same = torch.dot(distance_label_same , distance.view(-1)) / torch.sum(distance_label_same)
		distance_diff = torch.dot(distance_label_diff , distance.view(-1)) / torch.sum(distance_label_diff)
		margin = 600
		loss = F.relu( distance_same - distance_diff + margin )

		# accuracy
		predicted_label = torch.where(distance.view(-1) < 660,  torch.ones(family_0.size()[0]).to("cuda"), torch.full((family_0.size()[0],), -1).to("cuda"))
		match = torch.where((distance_label * predicted_label) == 1, torch.ones(family_0.size()[0]).to("cuda"), torch.zeros(family_0.size()[0]).to("cuda"))
		correct = torch.sum(match)
		return loss, correct

	def train_MUL(self, z0_list, z1_list, common_index_0, common_index_1, seq_len_0, seq_len_1):   
		bert_scores, _ = self.match(z0_list, z1_list)
		loss = 0.0
		for i, bert_score in enumerate(bert_scores):
			loss += self.structural_learning(bert_score, common_index_0[i], seq_len_0[i], common_index_1[i], seq_len_1[i])
		return loss
	
	def test_align(self, low_seq_0, low_seq_1, z0_list, z1_list, common_index_0, common_index_1, seq_len_0, seq_len_1, show_aln):
		bert_scores, _ = self.match(z0_list, z1_list)
		sequence_a = num_to_base(low_seq_0)
		sequence_b = num_to_base(low_seq_1)
		len_pred_match = 0
		len_ref_match = 0
		len_TP = 0
		for i, bert_score in enumerate(bert_scores):
			bert_score = torch.flatten(bert_score.T).tolist()
			x = 1 if show_aln == True else 0
			common_index_A_B = Aln_C.global_aln(bert_score, [0] * len(bert_score), [0] * len(bert_score), sequence_a[i], sequence_b[i], seq_len_0[i], seq_len_1[i], self.config.gap_opening, self.config.gap_extension, x, 0)
			common_index_A_B = torch.tensor(common_index_A_B).to(self.device).view(2, -1)
			len_pred_match += int(torch.sum(common_index_A_B[0]))
			len_ref_match += int(torch.sum(common_index_0[i])) 
			a = torch.flatten( (common_index_A_B[0] == 1).nonzero()*10000 + (common_index_A_B[1] == 1).nonzero()).tolist()
			b = torch.flatten( (common_index_0[i] == 1).nonzero()*10000 + (common_index_1[i] == 1).nonzero()).tolist()
			len_TP += len(set(a) & set(b))
		return len_TP, len_pred_match, len_ref_match 
	
	def structural_learning(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
		reference_alignment_score = self.calc_reference_alignment_score(bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1)
		prediction_alignment_score, _ = self.calc_prediction_alignment_score(bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1)
		return prediction_alignment_score - reference_alignment_score

	def match_bert_score(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
		index_0 = (common_index_0 == 1).nonzero()
		index_1 = (common_index_1 == 1).nonzero()
		index = torch.cat([index_0, index_1], axis=1).T
		index = tuple(index)
		omega = bert_score[index]
		return omega

	def margin_matrix(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
		index_0 = (common_index_0 == 1).nonzero()
		index_1 = (common_index_1 == 1).nonzero()
		index = torch.cat([index_0, index_1], axis=1).T
		index = tuple(index)
		margin_mat_FP = torch.ones(bert_score.shape).to(self.device) * self.config.margin_FP
		margin_mat_FP[index] = 0.0
		margin_mat_FN = torch.zeros(bert_score.shape).to(self.device)
		margin_mat_FN[index] = self.config.margin_FN
		return margin_mat_FP, margin_mat_FN

	def margin_score(self, common_index_A_B, common_index_0, common_index_1):
		len_pred_match = int(torch.sum(common_index_A_B[0]))
		len_ref_match = int(torch.sum(common_index_0)) 
		a = torch.flatten( (common_index_A_B[0] == 1).nonzero()*10000 + (common_index_A_B[1] == 1).nonzero()).tolist()
		b = torch.flatten( (common_index_0 == 1).nonzero()*10000 + (common_index_1 == 1).nonzero()).tolist()
		len_TP = len(set(a) & set(b))
		len_FP = len_pred_match - len_TP
		len_FN = len_ref_match - len_TP
		return len_FP * self.config.margin_FP + len_FN * self.config.margin_FN

	def calc_reference_alignment_score(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
		reference_alignment_score = 0
		reference_alignment_score += self.gapscore(common_index_0, seq_len_0)
		reference_alignment_score += self.gapscore(common_index_1, seq_len_1)
		omega = self.match_bert_score(bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1)
		reference_alignment_score += torch.sum(omega)
		return reference_alignment_score

	def calc_prediction_alignment_score(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
		sequence_a = "N"*seq_len_0
		sequence_b = "N"*seq_len_1
		margin_mat_FP, margin_mat_FN = self.margin_matrix(bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1)
		common_index_A_B = Aln_C.global_aln(torch.flatten(bert_score.T).tolist(), torch.flatten(margin_mat_FP.T).tolist(), torch.flatten(margin_mat_FN.T).tolist(), sequence_a, sequence_b, seq_len_0, seq_len_1, self.config.gap_opening, self.config.gap_extension, 0, 0)
		common_index_A_B = torch.tensor(common_index_A_B).to(self.device).view(2, -1)
		prediction_alignment_score = self.calc_reference_alignment_score(bert_score, common_index_A_B[0], seq_len_0, common_index_A_B[1], seq_len_1) + self.margin_score(common_index_A_B, common_index_0, common_index_1)
		return prediction_alignment_score, common_index_A_B

	def gapscore(self, index, seq_len):
		seq_len = int(seq_len)
		index = index[:seq_len].to('cpu').detach().numpy().copy()
		all_zeros = seq_len - np.count_nonzero(index)
		extend_zeros = seq_len - np.count_nonzero(np.insert(index[:-1], 0, 1) + index)
		open_zeros = all_zeros - extend_zeros
		return (-1*self.config.gap_opening + -1*self.config.gap_extension ) * open_zeros + -1*self.config.gap_extension * extend_zeros

	def match(self,z0_list, z1_list):
		match_scores = []
		logits = []
		for z0, z1 in zip(z0_list, z1_list):
			# cos similarity
			match_score = nn.CosineSimilarity(dim=2, eps=1e-6)(z0.unsqueeze(1).repeat(1, z1.shape[0],1) , z1.unsqueeze(0).repeat(z0.shape[0], 1,1))
			s = 1.3 * match_score
			# L1 distance
			# s = -torch.sum(torch.abs(z0.unsqueeze(1)-z1), -1)
			# match_score = torch.exp(s)
			
			# soft align
			a, b = F.softmax(s, 1), F.softmax(s, 0)
			c = a + b - a*b
			c = torch.sum(c*s)/torch.sum(c)
			match_scores.append(match_score)
			logits.append(c.view(-1))
		logits = torch.stack(logits, 0)
		return match_scores, logits

	def em(self, h, lengths):
		# get representations with different lengths from the collated single matrix
		e = [None] * len(lengths)
		for i in range(len(lengths)):
			e[i] = h[i, :lengths[i]]
		return e