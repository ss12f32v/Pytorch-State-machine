import numpy as np
import random 

state_number =5 
batch_size = 20 

state_dict={'A':0,'B':1,'C':2,'D':3,'E':4}
		
def one_hot_encoding( length, index):
	one_hot = np.zeros(length)
	one_hot[int(index)]=int(1)
	return  one_hot


class Parser(object):
	def split_data(self, sequence):
		splited = [x.strip() for x in sequence.split(',')]
		return splited

	def procedure(self, x):
		x = self.split_data(x)
		x[0] = int(x[0])
		x[1] = one_hot_encoding(length=len(state_dict) ,index=state_dict[x[1]])
		x[2] = int(x[2])
		return x 

	def mini_batch(self, batch_size=batch_size):
		for i in range (batch_size):
			myline =random.choice(dataset)
			x = self.procedure(myline)

		print(x[0],x[1],x[2])

if __name__== '__main__':
	dataset = open("Data.txt",'r',encoding='utf-8').read().splitlines()
	# vocab = Vocabulary();
	parse = Parser()
	# print("test one hot encoding array")
	# a = vocab.state_dict['B']
	# print(vocab.one_hot_encoding(5,a))
	print("test total procedure")
	x = "1,A,2"
	parse.mini_batch()
	print ( parse.procedure(x))




