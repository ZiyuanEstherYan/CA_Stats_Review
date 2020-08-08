import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

uber = pd.read_pickle("../data/uber.pkl")

# review the data
uber.describe()
pd.crosstab(uber["ccon"], uber["fs"])
sns.pairplot(uber)

fig, axs = plt.subplots(nrows=2, ncols=2)
sns.boxplot(x="fs", y="price", data=uber, ax=axs[0, 0])
sns.boxplot(x="fs", y="time", data=uber, ax=axs[0, 1])
sns.boxplot(x="ccon", y="price", data=uber, ax=axs[1, 0])
sns.boxplot(x="ccon", y="time", data=uber, ax=axs[1, 1])

sns.regplot(x="time", y="price", data=uber)

# run linear regression with only time in the model
mod = smf.ols(formula="price ~ time", data=uber).fit()
mod.summary()

# adding the friday/saturday effect
mod = smf.ols(formula="price ~ time + fs", data=uber).fit()
mod.summary()

# adding time:fs interaction effect
mod = smf.ols(formula="price ~ time + fs + time:fs", data=uber).fit()
mod.summary()

# adding the comi-con effect
mod = smf.ols(formula="price ~ time + fs + ccon", data=uber).fit()
mod.summary()

# adding the ccon:fs interaction effect
mod = smf.ols(formula="price ~ time + fs + ccon + fs:ccon", data=uber).fit()
mod.summary()

# full model
mod = smf.ols(formula="price ~ time + fs + ccon + time:fs + fs:ccon", data=uber).fit()
mod.summary()

# neural network
# start with one-hot encoding
uber_oh = pd.get_dummies(uber, drop_first=True)

# standardize the data
scaler = StandardScaler()
Xs = scaler.fit_transform(uber_oh.drop(columns=["price"]))
y = scaler.fit_transform(uber_oh[["price"]]).ravel()

# Note: for large neural networks I recommend using
# relu as the activation function. However, in smaller
# nets, "tanh" offers additiona flexibility with fewer
# hidden nodes and layers
# neural network with one node in the hidden layer
clf = MLPRegressor(
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

result = permutation_importance(clf, Xs, y, n_repeats=10, random_state=1234, n_jobs=-1)
idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result.importances[idx].T, vert=False, labels=uber_oh.columns[idx + 1])
ax.set_title("Permutation Importances")
fig.tight_layout()
plt.xlim(0, 1)
plt.show()

# Note: for large neural networks I recommend using
# relu as the activation function. However, in smaller
# nets, "tanh" offers additiona flexibility with fewer
# hidden nodes and layers
# neural network with two nodes in the hidden layer
clf = MLPRegressor(
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

result = permutation_importance(clf, Xs, y, n_repeats=10, random_state=1234, n_jobs=-1)
idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result.importances[idx].T, vert=False, labels=uber_oh.columns[idx + 1])
ax.set_title("Permutation Importances")
fig.tight_layout()
plt.xlim(0, 1)
plt.show()
