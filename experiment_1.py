from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

alpha = 0.001
iter = 100
accuracy = 0.001

m = 506
m_train = 203
d_test = 203
properti = 13
theta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

iter_num = [1] * iter
loss_train = [1] * iter
loss_test = [1] * iter


def get_data():
    data = load_svmlight_file(r"C:\Users\israr\Desktop\hotmodel\DATA\housing_scale.txt", n_features=13)

    return data[0], data[1]


X, y = get_data()
X = X.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=43)
print(X_train)
print(y_train)


def hypothesis(x):
    result = theta[0]
    for i in range(0, properti):
        result = result + theta[i + 1] * x[i]
    return result


def loss(m, X, y):
    sum = 0
    for i in range(0, m):
        sum = sum + (hypothesis(X[i]) - y[i]) ** 2
    sum = sum / (2 * m)
    return sum


def derivative(j, m, X, y):
    sum = 0

    if (j == 0):
        for i in range(0, m):
            sum = sum + (hypothesis(X[i]) - y[i])
    else:
        for i in range(0, m):
            sum = sum + (hypothesis(X[i]) - y[i]) * X[i][j - 1]
        sum = sum / m

    return sum


def train():
    for i in range(0, iter):

        for j in range(0, properti + 1):
            theta[j] = theta[j] - alpha * derivative(j, m_train, X_train, y_train)

        iter_num[i] = i;
        loss_train[i] = loss(m_train, X_train, y_train);
        loss_test[i] = loss(d_test, X_test, y_test);


def information():
    print("loss on train:", loss_train)
    print("loss on  test", loss_test)


train()
information()

fig, ax = plt.subplots()
ax.plot(iter_num, loss_train, color='m', label='loss of train')
ax.plot(iter_num, loss_test, color='c', label='loss of test')

ax.set_xlabel('Iteration times')
ax.set_ylabel('loss')
plt.xticks(iter_num, rotation=0)
plt.show()

