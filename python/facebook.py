import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from pyrsm import or_ci, gains_plot

facebook = pd.read_pickle("../data/facebook.pkl")
facebook.dtypes
facebook["click_yes"] = (facebook["click"] == "yes").astype(int)
mod = smf.glm(
    formula="click_yes ~ age + gender + ad",
    family=Binomial(link=logit()),
    data=facebook,
).fit()
mod.summary()
or_ci(mod)
facebook["logit"] = mod.predict(facebook)

# Fit neural network
# standardize the data
scaler = StandardScaler()
cols = np.array(["age", "gender_female", "ad_B"])
Xs = pd.get_dummies(facebook, drop_first=True)[cols]
Xs = scaler.fit_transform(Xs)
y = facebook["click_yes"]

# Note: for large neural networks I recommend using
# relu as the activation function. However, in smaller
# nets, "tanh" offers additiona flexibility with fewer
# hidden nodes and layers
clf = MLPClassifier(
    activation="tanh",
    # activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    alpha=0.01,
    hidden_layer_sizes=(1,),
    random_state=1234,
    max_iter=1000,
)
clf.fit(Xs, y)
facebook["nn1"] = clf.predict_proba(Xs)[:, 1]

result = permutation_importance(
    clf, Xs, y, scoring="roc_auc", n_repeats=10, random_state=1234, n_jobs=-1
)
idx = result.importances_mean.argsort()
fig, ax = plt.subplots()
ax.boxplot(result.importances[idx].T, vert=False, labels=cols[idx])
ax.set_title("Permutation Importances")
fig.tight_layout()
plt.xlim(0,)
plt.show()

# Note: for large neural networks I recommend using
# relu as the activation function. However, in smaller
# nets, "tanh" offers additiona flexibility with fewer
# hidden nodes and layers
clf = MLPClassifier(
    activation="tanh",
    # activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    alpha=0.01,
    hidden_layer_sizes=(2,),
    random_state=1234,
    max_iter=1000,
)
clf.fit(Xs, y)
# print(clf.coefs_)
facebook["nn2"] = clf.predict_proba(Xs)[:, 1]

result = permutation_importance(
    clf, Xs, y, scoring="roc_auc", n_repeats=10, random_state=1234, n_jobs=-1
)
idx = result.importances_mean.argsort()
fig, ax = plt.subplots()
ax.boxplot(result.importances[idx].T, vert=False, labels=cols[idx])
ax.set_title("Permutation Importances")
fig.tight_layout()
plt.xlim(0,)
plt.show()

# augmenting the logistic regression model
mod_int = smf.glm(
    formula="click_yes ~ age + gender + ad + gender:ad",
    family=Binomial(link=logit()),
    data=facebook,
).fit()
mod_int.summary()
or_ci(mod_int)
facebook["logit_int"] = mod_int.predict(facebook)
gains_plot(facebook, "click", "yes", ["logit", "logit_int", "nn1", "nn2"])
