from sklearn import datasets
from sklearn.model_selection import train_test_split

import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, 'class')
sys.path.append(mymodule_dir)

from mbgd import MiniBatchGradientDescent

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)

#create MiniBatchGradientDescent untuk dataset iris
mbgddi = MiniBatchGradientDescent(max_iter = 200, batch_size = 10, error_threshold = 0.0001, learning_rate = 0.5, n_layer = 3, n_neuron_per_layer = [4, 2, 3], activation_function_name_per_layer = ['none', 'sigmoid', 'sigmoid', 'sigmoid'])
mbgddi.set_input_data(X_train)
target = []
for i in range (len(y_train)):
    targetTemp = []
    for j in range (3):
        if (j == y_train[i]):
            targetTemp.append(1)
        else:
            targetTemp.append(0)
        target.append(targetTemp)
mbgddi.set_target(target)

mbgddi.information()

mbgddi.train()

mbgddi.information()
