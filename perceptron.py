import matplotlib.pyplot as plt 
import numpy as np

# faire des prediction par des weights
def preceptron(data, weights, n):
    for row in data:
        output = sum(row[:-1]*weights)
        if activation(output) != row[-1]:
            weights = weights + n*(row[-1] - activation(output))*row[:-1]
        else:
            continue
    return weights

def preceptron_test(data_test, weights):
    for row in data_test:
        output = sum(row[:-1]*weights)
        print(f"Expected={row[-1]}, Predicted={activation(output)}")	
        
def activation(output):
    return 1.0 if output >= 0.0 else -1.0


# data 			   x1    x2   x3    x4   y
dataset = np.array([[0.0, 0.0, 0.0, 0.0, 1.0],
				   [0.0, 0.0, 0.0, 1.0, -1.0],
				   [0.0, 0.0, 1.0, 0.0, 1.0],
				   [0.0, 0.0, 1.0, 1.0, -1.0],
				   [0.0, 1.0, 0.0, 0.0, 1.0],
				   [0.0, 1.0, 0.0, 1.0, -1.0],
				   [0.0, 1.0, 1.0, 0.0, 1.0],
				   [0.0, 1.0, 1.0, 1.0, -1.0],
				   [1.0, 0.0, 0.0, 0.0, 1.0],
				   [0.0, 0.0, 1.0, 0.0, -1.0]
				   ])
#test data
test_data = np.array([[1.0, 1.0, 0.0, 0.0, 1.0],
					 [1.0, 1.0, 0.0, 1.0, -1.0],
					 [1.0, 1.0, 1.0, 0.0, 1.0],
					 [1.0, 1.0, 1.0, 1.0, -1.0],
					]) 
n = 1
weights = [-0.1, 0.206, -0.23, 1]
preceptron_test(test_data, preceptron(dataset, weights, n))

print(preceptron(dataset, weights, n))
