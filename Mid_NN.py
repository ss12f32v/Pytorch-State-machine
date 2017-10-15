import random
import torch
import torch.nn as nn
import Data_helper as dp 
from torch.autograd import Variable
import torch.nn.functional as F

state_number =5 
embedding_dim = 100
embedding_size = 200
hidden_size =100
class MainNN_Model(nn.Module):
	def __init__(self, StateEncoder, TransitionEncoder):
		super(MainNN_Model, self).__init__()
		self.StateEncoder = StateEncoder
		self.TransitionEncoder = TransitionEncoder
		self.hidden_size = hidden_size
		self.linear1 = nn.Linear(embedding_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, state_number)
	def forward(self,inputs):
		x_0, x_1 = inputs
		StateEncoder_output =self.StateEncoder(x_0)
		TransitionEncoder_output =self.TransitionEncoder(x_1)

		MainNN_input = torch.cat((StateEncoder_output,TransitionEncoder_output),1)
		# print(MainNN_input.size())

		hidden_vector = F.relu(self.linear1(MainNN_input))
		out = self.linear2(hidden_vector)
		log_probs = F.log_softmax(out)
		return log_probs






