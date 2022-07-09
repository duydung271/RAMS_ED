import numpy as np
import transformers

mean_matrix = np.full((3),100)
transform_matrix = np.zeros((5,4))
print(mean_matrix)
# print(transform_matrix)
# transform_matrix[2:3,1] = mean_matrix
print(transform_matrix[2:3+1,1].shape)