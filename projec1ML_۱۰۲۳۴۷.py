from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
digit = load_digits()
print(digit.DESCR)
print(digit.data.shape)
print(digit.target.shape)
print(digit.images.shape)
print(digit.feature_names)

def image1(image):
    x = digit.images[image]
    plt.gray()
    plt.imshow(x)
    plt.show()


# Now We want do preprocessing our Code
x_train, x_test, y_train, y_test = train_test_split(digit.data,digit.target,test_size=0.3)
print(x_train)
# We needed to normalyaize our code because our code become in range (0,1)

scaler = MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])
print("<=====================>")
print(x_test[0])
# Preformance metrics

def calculate_matrics(y_train,y_test,y_pred_train,y_pred_test):
    acc_train = accuracy_score(y_true = y_train,y_pred = y_pred_train)
    acc_test = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    re_test = recall_score(y_true=y_test, y_pred=y_pred_test,average = 'weighted')
    pre_test = precision_score(y_true=y_test, y_pred=y_pred_test,average = 'weighted')
    print("acc_train=",acc_train)
    print("acc_test=",acc_test)
    print("re_test=",re_test)
    print("pre_test=",pre_test)
    return acc_train,acc_test,re_test,pre_test

rf = RandomForestClassifier(max_depth = 128,n_estimators = 300)
rf_ = rf.fit(x_train,y_train)
print("Ran======dom======Forest ==> ",rf_)
y_pred_train_rf = rf.predict(x_train)
y_pred_test_rf = rf.predict(x_test)
acc_train_rf,acc_test_rf,re_test_rf,pre_test_rf = calculate_matrics(y_train,y_test,y_pred_train_rf,y_pred_test_rf)


svm = SVC(kernel = 'poly',C = 2,gamma = 1)
### These are some parameter for [Kernal ==> ('poly', 'linear', 'rbf', 'sigmoid', 'precomputed')]
svm = svm.fit(x_train,y_train)
print("S======V======M==> ",svm)
y_pred_train_svm = svm.predict(x_train)
y_pred_test_svm = svm.predict(x_test)
acc_train_svm,acc_test_svm,re_test_svm,pre_test_svm = calculate_matrics(y_train,y_test,y_pred_train_svm,y_pred_test_svm)


mlp = MLPClassifier(hidden_layer_sizes = 256, max_iter = 10000,batch_size=65,alpha = 0.4,activation = 'relu',learning_rate = 'adaptive')
mlp = mlp.fit(x_train,y_train)
print("M======L======P==> \n",mlp)
y_pred_train_mlp = mlp.predict(x_train)
y_pred_test_mlp = mlp.predict(x_test)
acc_train_mlp,acc_test_mlp,re_test_mlp,pre_test_mlp=calculate_matrics(y_train,y_test,y_pred_train_mlp,y_pred_test_mlp)
