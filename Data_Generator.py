import numpy as np
import random


Trans1 = '0,A,1'
Trans2 = '1,B,2'
Trans3 = '2,C,3'
Trans4 = '1,D,4'
Trans5 = '4,E,2'
Set = [Trans1,Trans2,Trans3,Trans4,Trans5]
file = open("Data.txt", "w", encoding="UTF-8")

# print (len(Set))
lines_flag  = 500
class generator():
	def generate2File(self, lines = lines_flag,gener_flag = None):
		if gener_flag == True:
			for i in range(lines):
				file.write(Set[random.randint(0,4)])
				file.write('\n')
	def generateLine(self):
		return Set[random.randint(0,4)]

if __name__== '__main__':
	x  =  generator()
	print (x.generateLine())
	x.generate2File(gener_flag=True)
