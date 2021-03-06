from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import chocolate as choco

def score_gbt(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    gbt = GradientBoostingClassifier(**params)
    gbt.fit(X_train, y_train)
    y_pred = gbt.predict(X_test)

    return -precision_score(y_test, y_pred), -recall_score(y_test, y_pred)

X, y = make_classification(n_samples=80000, random_state=1)

conn = choco.SQLiteConnection(url="sqlite:///db.db")
s = {"learning_rate" : choco.uniform(0.001, 0.1),
     "n_estimators"  : choco.quantized_uniform(25, 525, 25),
     "max_depth"     : choco.quantized_uniform(2, 10, 2),
     "subsample"     : choco.quantized_uniform(0.7, 1.05, 0.05)}

sampler = choco.MOCMAES(conn, s, mu=5)
token, params = sampler.next()
loss = score_gbt(X, y, params)
sampler.update(token, loss)

conn = choco.SQLiteConnection(url="sqlite:///db.db")
results = conn.results_as_dataframe()
losses = results.as_matrix(("_loss_0", "_loss_1"))
first_front = choco.mo.argsortNondominated(losses, len(losses), first_front_only=True)

plt.scatter(losses[:, 0], losses[:, 1], label="All candidates")
plt.scatter(losses[first_front, 0], losses[first_front, 1], label="Optimal candidates")
plt.xlabel("precision")
plt.ylabel("recall")
plt.legend()

plt.show()