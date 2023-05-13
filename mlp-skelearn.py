from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)

#Perbandingan dengan MLP Skelearn untuk dataset iris
clf = MLPClassifier(
    hidden_layer_sizes=(2),
    activation="logistic", 
    solver="sgd", 
    batch_size=10, 
    learning_rate_init=0.5, 
    learning_rate="constant", 
    max_iter=200
)
clf.fit(X_train, y_train)
weightArr = clf.coefs_
biasArr = clf.intercepts_

print("Jumlah layer: {}".format(clf.n_layers_))
print("Jumlah iterasi: {}".format(clf.n_iter_))
print("Jumlah fitur: {}".format(clf.n_features_in_))
print("Jumlah output: {}".format(clf.n_outputs_))

# Print array weight untuk setiap layer
for i in range(len(biasArr)):
	if ( i+1 != len(biasArr)):
		print("")
		print(f"Pada hidden layer yang ke-{i+1} terdapat")
		print(f"Bias: {biasArr[i]}")
		print(f"Weight: {weightArr[i]}")
	else :
		print("")
		print(f"Pada output layer terdapat")
		print(f"Bias: {biasArr[i]}")
		print(f"Weight: {weightArr[i]}")