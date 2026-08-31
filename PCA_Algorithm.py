from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
data = load_iris()
x = data.data
y = data.target
print(x)
print(y)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
pca = PCA(n_components = 2)
x_pca = pca.fit_transform(x_scaled)

print("Orginal Shape: ",x.shape)
print("PCA shape: ",x_pca.shape)
print("After the PCA Algorithm and Scaler: ",x_pca)

print("Explaned variance ratio:")
print(pca.explained_variance_ratio_)
x_train,x_test,y_train,y_test = train_test_split(x_pca,y,test_size=0.2)
lr = LogisticRegression().fit(x_train,y_train)
print(lr)
y_pred_test= lr.predict(x_test)
y_pred_train = lr.predict(x_train)
print(y_pred_test)
acc_test = accuracy_score(y_test,y_pred_test)
acc_train = accuracy_score(y_train,y_pred_train)
print(f"Accuracy Test: {acc_test:.02f}")
print(f"Accuracy Train: {acc_train:.02f}")
