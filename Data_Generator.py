
import torch
import numpy as np
import random
from torch.autograd import Variable

Trans1 = '0,A,1'
Trans2 = '1,B,2'
Trans3 = '2,C,3'
Trans4 = '1,D,4'
Trans5 = '4,F,2'
Set = [Trans1,Trans2,Trans3,Trans4,Trans5]
file = open("Data.txt", "w", encoding="UTF-8")

print (len(Set))
lines  = 500
for i in range(lines):
	file.write(Set[random.randint(0,4)])
	file.write('\n')

