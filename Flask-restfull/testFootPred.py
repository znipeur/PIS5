import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\HP\\Desktop\\PI\\Flask-restfull\\E0.csv')
X = dataset.iloc[:, [3, 4]].values
y = dataset.iloc[:, 7].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
labelencoder_X.fit(X[0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotIncoder = OneHotEncoder(categorical_features=[0, 1])
X = onehotIncoder.fit_transform(X).toarray()
X[:, 0:20] = X[:, 1:21]
X[:, 20:38] = X[:, 22:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train[:, 0:38], y_train)

y_pred = knn.predict(X_test[:, 0:38])

# prodctedMatch = onehotIncoder.fit_transform([labelencoder_X.fit_transform('Liverpool'), labelencoder_X.fit_transform('Man City')]).toarray()

# TODO fix the prediction of one match .
# y_pred2 = knn.predict([prodctedMatch[:, 1:21], prodctedMatch[:, 22:]])

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(cm)
total_correct = 0
for i in range(0, 3):
    total_correct = total_correct + cm[i][i]
print(total_correct)

from matplotlib.colors import ListedColormap

X_set, y_set = X, y
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
#              cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i),
#                 label=j)
# plt.legend()
# plt.show()
