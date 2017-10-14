import torch
import numpy as np

from torch.autograd import Variable
state_number =5 
batch_size = 20 
class Vocabulary(object):
	def __init__(self):
		self.state_dict={'A':0,'B':1,'C':2,'D':3,'E':4}
		
	def one_hot_encoding(self, length, index):
		one_hot = np.zeros(length)
		one_hot[int(index)]=int(1)
		return  one_hot


class Parser(object):
	def split_data(self, sequence):
		splited = [x.strip() for x in sequence.split(',')]
		return splited

def procedure(x):
	x = parse.split_data(x)
	x[0] = vocab.one_hot_encoding(state_number,x[0])
	x[1] = vocab.one_hot_encoding(len(vocab.state_dict),vocab.state_dict[x[1]])
	x[2] = vocab.one_hot_encoding(state_number,x[2])
	return x 

def mini_batch(batch_size,original_index):
	for i in range(batch_size):
		line = dataset.readline()


if __name__== '__main__':
	dataset =  open("Data.txt",'r',encoding='utf-8') 
	vocab = Vocabulary();
	parse = Parser()
	# print("test one hot encoding array")
	# a = vocab.state_dict['B']
	# print(vocab.one_hot_encoding(5,a))
	print("test total procedure")
	x = "1,A,2"
	print (x = procedure(x))




