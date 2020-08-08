import pandas as pd
import statsmodels.formula.api as smf
from pyrsm import describe, expand_grid, levels_list

# load data
did = pd.read_pickle("../data/did_sales_force.pkl")

# review the data
describe(did)
did.describe()
did.dtypes
did["region"]
did["period"]

# create a pivot table
pd.crosstab(did["region"], did["period"], values=did["CLV"], aggfunc="mean")

# run linear regression
mod = smf.ols(formula="CLV ~ region + period + region:period", data=did).fit()
mod.summary()

# generate predictions
dct = levels_list(did[["region", "period"]])
pred_data = expand_grid(dct)
pred_data["prediction"] = mod.predict(pred_data)
pred_data
