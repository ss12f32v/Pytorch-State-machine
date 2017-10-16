import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Data_helper as dp
import State_and_Transition_encoding 
import Mid_NN

epoch = 2000
state_number =5 
embedding_dim = 100
batch_size = 20 
checkpoint_name="model_checkpoint/model_parameter.pt"

parser = dp.Parser()

losses = []

loss_function = nn.NLLLoss()

StateEncodeModel = State_and_Transition_encoding.StateEncodeModel(state_number, batch_size, embedding_dim)
TransistionEncodelModel = State_and_Transition_encoding .TransistionEncodelModel(state_number, batch_size, embedding_dim)

model = Mid_NN.MainNN_Model(StateEncodeModel, TransistionEncodelModel)
# model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001)
def main():
	for i in range(epoch):
		total_loss = torch.Tensor([0])
		x,y,z=parser.mini_batch()
		inputs = (x,y)
		tagets = z
		inputs = autograd.Variable(torch.LongTensor(inputs))

		model.zero_grad()
		log_probs = model(inputs)
		loss = loss_function(log_probs, autograd.Variable(torch.LongTensor(tagets)))

		
		# print(autograd.Variable(torch.LongTensor(tagets)))
		loss.backward()
		optimizer.step()

		# print("Loss: ",loss)
	torch.save(model.state_dict(),checkpoint_name)
	print("Model has been saved as %s.\n" % checkpoint_name)

	print("Testing Model Performance.....")

	model.eval()
	eval_loss = 0
	eval_acc = 0
	x,y,z=parser.mini_batch(batch_size= 20)
	print("Inputs :", x,y,z)
	inputs = (x,y)
	tagets = z
	inputs = autograd.Variable(torch.LongTensor(inputs))
	log_probs = model(inputs)
	print("Prdict : ",torch.max(log_probs, 1))




if __name__== '__main__':
	main()