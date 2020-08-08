import pandas as pd
import statsmodels.formula.api as smf
from pyrsm import describe, vif

# load data
price_sales = pd.read_pickle("../data/price_sales.pkl")
diamonds = pd.read_pickle("../data/diamonds.pkl")
click = pd.read_pickle("../data/click.pkl")
click

# review the data
describe(diamonds)
describe(price_sales)
describe(click)

# run linear regression on price_sales data
smf.ols(formula="sales1 ~ price", data=price_sales).fit().summary()

# run linear regression on diamonds data
smf.ols(formula="price ~ clarity", data=diamonds).fit().summary()
smf.ols(formula="price ~ carat + clarity", data=diamonds).fit().summary()

# correlations between carat and clarity
pd.get_dummies(diamonds[["carat", "clarity"]]).corr()

# run linear regression on click data
smf.ols(formula="sales ~ advertising", data=click).fit().summary()
smf.ols(formula="sales ~ salesreps", data=click).fit().summary()

mod = smf.ols(formula="sales ~ advertising + salesreps", data=click).fit()
mod.fit.summary()

# vif requires a model that has not yet been fitted
vif(mod)
click.corr()
