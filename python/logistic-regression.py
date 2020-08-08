import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit
from pyrsm import describe, or_ci, or_plot, ifelse, scale_df

# load data
dvd = pd.read_pickle("../data/dvd.pkl")
dvd["buy_yes"] = (dvd["buy"] == "yes").astype(int)
dvd

# review the data
describe(dvd)

# Logistic regression with only the `coupon` variable
lr = smf.glm(formula="buy_yes ~ coupon", family=Binomial(link=logit()), data=dvd,).fit()
lr.summary()
or_ci(lr)

# Logistic regression with `coupon`, `purch`, and `last` variables
lr = smf.glm(
    formula="buy_yes ~ coupon + purch + last", family=Binomial(link=logit()), data=dvd,
).fit()
lr.summary()
or_ci(lr)
or_plot(lr)

# Logistic regression with standardized ORs and Importance values
dvd_st = dvd.copy()
evar = ["coupon", "purch", "last"]
dvd_st[evar] = scale_df(dvd[evar])
lr = smf.glm(
    formula="buy_yes ~ coupon + purch + last",
    family=Binomial(link=logit()),
    data=dvd_st,
).fit()
lr.summary()
ORs = or_ci(lr)
ORs["Importance"] = ifelse(ORs["OR"] > 1, ORs["OR"], 1 / ORs["OR"])
ORs.sort_values(by="Importance", ascending=False)
or_plot(lr)

# summary statistics
dvd.mean()
dvd.std()
