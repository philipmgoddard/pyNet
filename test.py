import numpy as np

dim1 = [2, 4, 6]
dim2 = [3, 6, 9]

print(dim1, dim2)


a_tmp = list()
for i,j in zip(dim1, dim2):
  a_tmp.append(np.empty([i, j]))



print(a_tmp)



print([np.empty(x) for x in zip(dim1, dim2)])
