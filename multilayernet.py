"""
multilayernet.py
~~~~~~~~~~~~~~~~
---example network---
      |o|-->|o|
|o|--> |     |
      |o|-->|o|-->|o| (2 features, 2 hidden layers, 1 output node)
|o|--> |     |
      |o|-->|o|

A simple python implementation of a multi layer neural net(2 hidden).
This implementation of the neural net is able to accept an arbitiary
number of features and examples.

---data structure---
the inputs are structured with the features being the row and examples shared in columns
example: if there is are 3 pieces of data (1, 2), (3, 4), (5, 6)
from the data provided there are 2 features and 3 examples
the data will be structures as so:
5 3 1 --> o
| | |
6 4 2 --> o
x = np.array([[1, 3, 5], [2, 4, 6]])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

x = np.array([[1]]) (1 example with 1 feature)
x = np.array([[1, 2]]) (2 examples with 1 feature)

x = np.array([[1], [1]) (1 example with 2 features)
x = np.array([[1, 1], [1, 1]) (2 examples with 2 features)
x = np.array([[1, 1, 1], [1, 1, 1]) (3 examples with 2 features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y = np.array([[1]]) (output for 1 example)
y = np.array([[1, 2]]) (output for 2 examples)

"""
import numpy as np
from progress.bar import ShadyBar

ITERATIONS = 5000

def sigmoid(x, der=False):
    if der is True:
        return sigmoid(x)*(1-sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))


def getshapeof(x):
    print(np.shape(x))

column1_ = []
column2_ = []
column3_ = []
column4_ = []
column5_ = []

# comment out this section for custom examples
with open('train_data.txt', 'r') as reader:
    lines = [line.rstrip('\n') for line in reader]
    for line in lines:
        column1, column2, column3, column4, column5 = line.split(',')
        column1_.append(float(column1))
        column2_.append(float(column2))
        column3_.append(float(column3))
        column4_.append(float(column4))
        column5_.append(float(column5))

    x = np.array([(column1_), (column2_), (column3_), (column4_)]).T
    y = np.array([(column5_)])

reader.close()


# custom examples
# x = np.array([[0, 1, 1, 0], [1, 1, 0, 0]]).T # input examples
# y = np.array([[1, 1, 1, 0]])

weight1 = np.random.random((x.shape[1], 8)) # weight between input and hidden layer 1
weight2 = np.random.random((8, 8)) # weight between hidden layer 1 and 2
weight3 = np.random.random((8, 1)) # weight between hidden layer 2 and output

bar = ShadyBar('Processing', max=ITERATIONS, suffix="%(percent)d%%")

# iterations
for i in range(ITERATIONS):

    derived_hidden_layer = sigmoid(np.dot(x, weight1))
    derived_hidden_layer2 = sigmoid(np.dot(derived_hidden_layer, weight2))

    output = sigmoid(np.dot(derived_hidden_layer2, weight3))
#     end of forward propagate
    error = np.square(np.mean(y.T - output))
    derived_error = y.T - output

    error_percent = (np.mean(output)-np.mean(y.T)) * 100

    scalar = derived_error * sigmoid(np.dot(derived_hidden_layer2, weight3), True) # this remains static throught the calculations

    weight3 += np.dot(scalar.T,derived_hidden_layer2).T * 0.01
    weight2 += np.dot(np.dot(scalar,weight3.T).T, sigmoid(np.dot(derived_hidden_layer, weight2),True) * sigmoid(np.dot(x, weight1),True)) * 0.01
    weight1 += np.dot((np.dot(scalar,weight3.T) * np.dot(sigmoid(np.dot(derived_hidden_layer, weight2),True),weight2) * sigmoid(np.dot(x, weight1),True)).T,x).T * 0.01

    # if i % 500 == 0:
    #     print()
    #     print("Error (%s'th iteration): %s" % (i,error))
    #     print()
    bar.next()

bar.finish()
print('%s error difference (<1 is better)' % np.abs(error_percent))
print(output)

column1_ = []
column2_ = []
column3_ = []
column4_ = []
column5_ = []
# test data
with open('test_data.txt', 'r') as testreader:
    lines = [line.rstrip('\n') for line in testreader]
    for line in lines:
        column1, column2, column3, column4, column5 = line.split(',')
        column1_.append(float(column1))
        column2_.append(float(column2))
        column3_.append(float(column3))
        column4_.append(float(column4))
        column5_.append(float(column5))

    test_x = np.array([(column1_), (column2_), (column3_), (column4_)]).T
    test_y = np.array([(column5_)])

testreader.close()

derived_hidden_layer_test = sigmoid(np.dot(test_x, weight1))
derived_hidden_layer2_test = sigmoid(np.dot(derived_hidden_layer_test, weight2))

test_output = sigmoid(np.dot(derived_hidden_layer2_test, weight3))

test_error = np.square(np.mean(test_y.T - test_output))
test_derived_error = test_y.T - test_output

test_error_percent = (np.mean(test_output)-np.mean(test_y.T)) * 100

print()
print('%s test error difference (<1 is better)' % np.abs(test_error_percent))
print(test_output)
