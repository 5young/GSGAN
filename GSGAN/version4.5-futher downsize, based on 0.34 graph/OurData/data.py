import numpy as np
import math

# data = np.loadtxt("GrQc.txt")
# data1 = data.reshape(2*data.shape[0]).tolist()
# data1 = list(set(data1))
# print(data1)
# print(len(data1))
# print(data1.index(3466))
# print(data1.index(24575))
# data3 = []

# file = open("GrQc_index.txt", "w")
# for i in range(data.shape[0]):
# 	n1 = data[i, 0]
# 	n2 = data[i, 1]

# 	data3.append([data1.index(n1), data1.index(n2)])
# 	# print(data1.index(n1), data1.index(n2))
# 	# input()

# for i in range(data.shape[0]):
# 	file.write(str(data3[i][0]))
# 	file.write(" ")
# 	file.write(str(data3[i][1]))
# 	file.write("\n")


data = np.loadtxt("GrQc_index.txt")
print(np.where(data==1))
