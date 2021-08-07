### распознование рукописных цифр mnist
import numpy as np
from urllib import request
import gzip
import pickle

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

#init()
X_train, y_train, X_test, y_test = load()
num_labels = len(y_train)
num_labels

# Теперь сначала делаем среднее 0 и стандартное отклонение 1

# делаем Мат Ожидание  = 0
X_train_1, X_test_1 = X_train - np.mean(X_train), X_test - np.mean(X_train)
np.min(X_train_1), np.max(X_train_1), np.min(X_test_1), np.max(X_test_1)

# Делаем чтобы стандартное отклонение было = 1

X_train_1, X_test_1 = X_train_1 / np.std(X_train_1), X_test_1 / np.std(X_train_1)

# one-hot encode
num_labels = len(y_train)
train_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    train_labels[i][y_train[i]] = 1

num_labels = len(y_test)
test_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    test_labels[i][y_test[i]] = 1

def calc_accuracy_model(model, test_set):
    return print(f'''The model validation accuracy is: {np.equal(np.argmax(model.forward(test_set), axis=1), y_test).sum() * 100.0 / test_set.shape[0]:.2f}%''')

from optimizers import SGD
# Подключаю библиотеки
from network import *
from train   import *
from optimizers import *
#
## Другая функция оптимизайии
#model = NeuralNetwork(
#    layers=[Dense(neurons=89, 
#                  activation=Sigmoid()),
#            Dense(neurons=10, 
#                  activation=Linear())],
#            loss = SoftmaxCrossEntropy(), 
#seed=20190119)
#
#optim = SGDMomentum(0.1, momentum=0.9)
#
#trainer = Trainer(model, SGDMomentum(0.1, momentum=0.9))
#trainer.fit(X_train_1, train_labels, X_test_1, test_labels,
#            epochs = 50,
#            eval_every = 1,
#            seed=20190119,
#            batch_size=60);
#
#calc_accuracy_model(model, X_test_1)# 92.15
#
#
#model = NeuralNetwork(
#    layers=[Dense(neurons=89, 
#                  activation=Tanh()),
#            Dense(neurons=10, 
#                  activation=Linear())],
#            loss = SoftmaxCrossEntropy(), 
#seed=20190119)
#
#optim = SGD(0.1)
#
#optim = SGDMomentum(0.1, momentum=0.9)
#
#trainer = Trainer(model, SGDMomentum(0.1, momentum=0.9))
#trainer.fit(X_train_1, train_labels, X_test_1, test_labels,
#            epochs = 50,
#            eval_every = 10,
#            seed=20190119,
#            batch_size=60);
#
#calc_accuracy_model(model, X_test_1) # 95 -> 40 EPOCH
#
#
#
#model = NeuralNetwork(
#    layers=[Dense(neurons=89, 
#                  activation=Tanh()),
#            Dense(neurons=10, 
#                  activation=Linear())],
#            loss = SoftmaxCrossEntropy(), 
#seed=20190119)
#
#
#optimizer = SGDMomentum(0.15, momentum=0.9, final_lr = 0.05, decay_type='linear')
#
#trainer = Trainer(model, optimizer)
#trainer.fit(X_train_1, train_labels, X_test_1, test_labels,
#            epochs = 50,
#            eval_every = 10,
#            seed=20190119,
#            batch_size=60);
#
#calc_accuracy_model(model, X_test_1) # 95.93	
#
#
### По
#
#model = NeuralNetwork(
#    layers=[Dense(neurons=89, 
#                  activation=Tanh(),
#                  weight_init="glorot"),
#            Dense(neurons=10, 
#                  activation=Linear(),
#                  weight_init="glorot")],
#            loss = SoftmaxCrossEntropy(), 
#seed=20190119)
#
#optimizer = SGDMomentum(0.15, momentum=0.9, final_lr = 0.05, decay_type='linear')
#
#trainer = Trainer(model, optimizer)
#trainer.fit(X_train_1, train_labels, X_test_1, test_labels,
#       epochs = 50,
#       eval_every = 10,
#       seed=20190119,
#           batch_size=60);
#
#calc_accuracy_model(model, X_test_1) #95.69 меньше ошибка

#model = NeuralNetwork(
#    layers=[Dense(neurons=89, 
#                  activation=Tanh(),
#                  weight_init="glorot",
#                  dropout=0.5),
#            Dense(neurons=10, 
#                  activation=Linear(),
#                  weight_init="glorot")],
#            loss = SoftmaxCrossEntropy(), 
#seed=20190119)
#
#trainer = Trainer(model, SGDMomentum(0.2, momentum=0.9, final_lr = 0.05, decay_type='exponential'))
#trainer.fit(X_train_1, train_labels, X_test_1, test_labels,
#       epochs = 50,
#       eval_every = 10,
#       seed=20190119,
#           batch_size=60,
#           early_stopping=True);
#
#calc_accuracy_model(model, X_test_1)



model = NeuralNetwork(
    layers=[Dense(neurons=178, 
                  activation=Tanh(),
                  weight_init="glorot",
                  dropout=0.8),
            Dense(neurons=46, 
                  activation=Tanh(),
                  weight_init="glorot",
                  dropout=0.8),
            Dense(neurons=10, 
                  activation=Linear(),
                  weight_init="glorot")],
            loss = SoftmaxCrossEntropy(), 
seed=20190119)

trainer = Trainer(model, SGDMomentum(0.2, momentum=0.9, final_lr = 0.05, decay_type='exponential'))
trainer.fit(X_train_1, train_labels, X_test_1, test_labels,
       epochs = 100,
       eval_every = 10,
       seed=20190119,
           batch_size=60,
           early_stopping=False);

calc_accuracy_model(model, X_test_1) #95.97