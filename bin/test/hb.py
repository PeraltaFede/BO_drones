from hummingbird.ml import convert
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from time import time
X, y = load_breast_cancer(return_X_y=True)
skl_model = RandomForestClassifier(n_estimators=1000, max_depth=7)
skl_model.fit(X, y)

t0 = time()
for i in range(50):
    pred = skl_model.predict(X)
print(time() - t0)
t0 = time()
model = convert(skl_model, 'torch')
tf = time()-t0
for i in range(50):
    pred_cpu_hb = model.predict(X)
print(time() - t0)

t0 = time()
model.to('cuda')
for i in range(50):
    pred_gpu_hb = model.predict(X)
print(time() - t0 + tf)
