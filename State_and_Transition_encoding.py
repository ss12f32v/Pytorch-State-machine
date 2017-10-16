
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Data_Generator import generator 

import Data_helper as dp
batch_size = 20
state_number = 5 
Transition_Number = 5 

class StateEncodeModel(nn.Module):
	def __init__(self, state_number, batch_size, embedding_dim=100):
		super(StateEncodeModel, self).__init__()
		self.embeddings = nn.Embedding(state_number, embedding_dim)
	def forward(self, inputs):
		embeds = self.embeddings(inputs)
		# print(embeds.size())
		return embeds.view(batch_size,-1)

class TransistionEncodelModel(nn.Module):
	def __init__(self, Transition_Number, batch_size, embedding_dim=100):
		super(TransistionEncodelModel, self).__init__()
		self.embeddings = nn.Embedding(Transition_Number, embedding_dim)
	def forward(self, inputs):
		embeds = self.embeddings(inputs)
		# print(embeds.size())
		return embeds.view(batch_size,-1)

if __name__== '__main__':
	print("Testing.....")
	x  =  generator()
	parser = dp.Parser()
	line = x.generateLine()
	line2 = x.generateLine()

	print(line)
	print (line2)
	line = parser.procedure(line)
	line2 = parser.procedure(line2)

	line = list(zip(line,line2))
	State_Model = StateEncodeModel(state_number=state_number, embedding_dim=100)
	print("input state index:",line[0])
	context_var = autograd.Variable(torch.LongTensor([line[0]]))
	print("context_var:", context_var)
	Vector = State_Model(context_var)
	print ("Embedding size : ",Vector.size())