# The slow way to generate an ROC curve
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from pyrsm import describe, or_ci

# load data
bbb = pd.read_pickle("../data/bbb.pkl")
bbb["buyer_yes"] = (bbb["buyer"] == "yes").astype(int)
bbb

# review the data
describe(bbb)
evar = [
    "gender",
    "last",
    "total",
    "child",
    "youth",
    "cook",
    "do_it",
    "reference",
    "art",
    "geog",
]

# Logistic regression with only the `coupon` variable
form = "buyer_yes ~ " + " + ".join(evar)
lr = smf.glm(formula=form, family=Binomial(link=logit()), data=bbb).fit()
lr.summary()
or_ci(lr)
bbb["pred_logit"] = lr.predict(bbb)

# Function to calculate the TPR and TNR at different trade-off
# values (and break-even values)


def slow_roc(outcome, pred, cost, margin):
    NA = np.nan
    tbl = pd.DataFrame(
        {
            "cost": cost,
            "margin": margin,
            "BE": cost / margin,
            "TP": NA,
            "FP": NA,
            "TN": NA,
            "FN": NA,
            "TNR": NA,
            "TPR": NA,
        }
    )
    for i in range(len(cost)):
        BEi = tbl.loc[i, "BE"]
        TP = np.where((pred > BEi) & (outcome == True), 1, 0).sum()
        FP = np.where((pred > BEi) & (outcome == False), 1, 0).sum()
        TN = np.where((pred <= BEi) & (outcome == False), 1, 0).sum()
        FN = np.where((pred <= BEi) & (outcome == True), 1, 0).sum()
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        tbl.iloc[i, 3:9] = [TP, FP, TN, FN, TNR, TPR]

    return tbl


# Creating the data for the ROC curve

outcome = bbb["buyer"] == "yes"
pred = bbb["pred_logit"]
cost = np.arange(0.0, 6.0, 0.05)
margin = 6
roc_data = slow_roc(outcome, pred, cost=np.arange(0.0, 6.0, 0.05), margin=6)

# Plotting the ROC curve
plt.cla()
fig = sns.lineplot(x="TNR", y="TPR", data=roc_data)
fig.set(ylabel="TPR (Sensitivity)", xlabel="TNR (Specificity)")
plt.gca().invert_xaxis()
fig.plot([0, 1], [0, 1], transform=fig.transAxes, linestyle="--")
plt.show()

# Calculating the TPR and TNR for the break-even point in the
# BBB case

bbb_to = slow_roc(outcome, pred, cost=np.array([0.5]), margin=6)

plt.cla()
fig = sns.lineplot(x="TNR", y="TPR", data=roc_data)
fig.set(ylabel="TPR (Sensitivity)", xlabel="TNR (Specificity)")
plt.gca().invert_xaxis()
plt.gca().set_aspect("equal", adjustable="box")
fig.plot([0, 1], [0, 1], transform=fig.transAxes, linestyle="--")
plt.scatter(bbb_to["TNR"], bbb_to["TPR"], s=50, color="red")
plt.annotate("cost: 0.5, BE: 0.083", (bbb_to["TNR"] - 0.03, bbb_to["TPR"]))
plt.show()

bbb_to = slow_roc(outcome, pred, cost=np.array([0]), margin=6)

plt.cla()
fig = sns.lineplot(x="TNR", y="TPR", data=roc_data)
fig.set(ylabel="TPR (Sensitivity)", xlabel="TNR (Specificity)")
plt.gca().invert_xaxis()
plt.gca().set_aspect("equal", adjustable="box")
fig.plot([0, 1], [0, 1], transform=fig.transAxes, linestyle="--")
plt.scatter(bbb_to["TNR"], bbb_to["TPR"], s=50, color="red")
plt.annotate("cost: 0, BE: 0.083", (bbb_to["TNR"] + 0.5, bbb_to["TPR"] + 0.02))
plt.show()

# Confirm plot with roc_curve from sklearn
fpr, tpr, thresholds = roc_curve(bbb["buyer_yes"], bbb["pred_logit"])
py_roc = pd.DataFrame({"FPR": fpr, "TPR": tpr})

# slow!
# plt.cla()
# fig = sns.lineplot(x="FPR", y="TPR", data=py_roc)
# fig.set(ylabel="TPR (Sensitivity)", xlabel="FPR")
# fig.plot([0, 1], [0, 1], transform=fig.transAxes, linestyle="--")
# plt.show()

# Probabilistic interpretation of AUC
# See https://www.alexejgossmann.com/auc/ for a very nice dicsussion
# Adapted from Alexej's code
s = 0
did_buy = np.where(outcome == True)[0]
did_not_buy = np.where(outcome == False)[0]
for i in did_buy:
    s = s + np.where(pred[i] > pred[did_not_buy], 1, 0).sum()
    s = s + np.where(pred[i] == pred[did_not_buy], 1, 0).sum() / 2

s / ((outcome == True).sum() * (outcome == False).sum())

# Lets compare that result to what we would get with a formal calculation:
auc(fpr, tpr)

# Lets try a sampling approach
pred_did_buy = pred[did_buy]
pred_did_not_buy = pred[did_not_buy]
nr = 2000
(np.random.choice(pred_did_buy, nr) > np.random.choice(pred_did_not_buy, nr)).mean()

# Lets do repeated simulation
rep = 100
s = np.array([np.nan] * rep)
for i in range(0, rep):
    s[i] = (
        np.random.choice(pred_did_buy, nr) > np.random.choice(pred_did_not_buy, nr)
    ).mean()

s.mean()
