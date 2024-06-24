import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Річний рівень дефолту
# з 01.20 до 12.22
npl_fact = [
    45.7,
    46.3,
    46.1,
    46.1,
    46.6,
    46.9,
    45.6,
    45.1,
    44.9,
    42.9,
    40.4,
    38.9,
    37.8,
    37.8,
    37.2,
    36.7,
    36.3,
    34.6,
    34,
    33.1,
    31.8,
    30.2,
    29.7,
    27.9,
    27.3,
    26.7,
    26.7,
    27.2,
    27.5,
    27.8,
    29.8,
    30.1,
    30.8,
    33.6,
    35.5,
    36.7,
    # 38.1,
    # 38.2,
    # 38.5,
]

npl_fact = list(np.array(npl_fact) / 100)
# ВВП
# з 01.18 до 12.18
gdp18 = [
    235004000000,
    235004000000,
    235004000000,
    270273000000,
    270273000000,
    270273000000,
    331616000000,
    331616000000,
    331616000000,
    349341000000,
    349341000000,
    349341000000,
]


# ВВП
# з 01.19 до 12.22
gdp = [
    273335333333,
    273335333333,
    273335333333,
    310819666666,
    310819666666,
    310819666666,
    370653333333,
    370653333333,
    370653333333,
    370923333333,
    370923333333,
    370923333333,
    286138333333,
    286138333333,
    286138333333,
    293307000000,
    293307000000,
    293307000000,
    391036666666,
    391036666666,
    391036666666,
    436860000000,
    436860000000,
    436860000000,
    339903333333,
    339903333333,
    339903333333,
    394103333333,
    394103333333,
    394103333333,
    504293333333,
    504293333333,
    504293333333,
    578646666666,
    578646666666,
    578646666666,
    363836666666,
    363836666666,
    363836666666,
    343633333333,
    343633333333,
    343633333333,
    487033333333,
    487033333333,
    487033333333,
    551870000000,
    551870000000,
    551870000000,
    # 420846666666,
    # 420846666666,
    # 420846666666,
]

# Дефлятор ВВП
# з 01.18 до 12.18
gdp18_deflator = [
    1.094,
    1.094,
    1.094,
    1.119,
    1.119,
    1.119,
    1.134,
    1.134,
    1.134,
    1.249,
    1.249,
    1.249,
]

# Дефлятор ВВП
# з 01.19 до 12.22
gdp_deflator = [
    1.06,
    1.06,
    1.06,
    1.061,
    1.061,
    1.061,
    1.056,
    1.056,
    1.056,
    1.118,
    1.118,
    1.118,
    1.038,
    1.038,
    1.038,
    1.037,
    1.037,
    1.037,
    1.058,
    1.058,
    1.058,
    1.187,
    1.187,
    1.187,
    1.131,
    1.131,
    1.131,
    1.168,
    1.168,
    1.168,
    1.182,
    1.182,
    1.182,
    1.281,
    1.281,
    1.281,
    1.134,
    1.134,
    1.134,
    1.241,
    1.241,
    1.241,
    1.256,
    1.256,
    1.256,
    1.344,
    1.344,
    1.344,
    # -16.1,
    # -16.1,
    # -16.1,
]

# Індекс споживчих цін
# з 01.19 до 12.22
cpi = [
    1,
    0.5,
    0.9,
    1,
    0.7,
    -0.5,
    -0.6,
    -0.3,
    0.7,
    0.7,
    0.1,
    -0.2,
    0.2,
    -0.3,
    0.8,
    0.8,
    0.3,
    0.2,
    -0.6,
    -0.2,
    0.5,
    1,
    1.3,
    0.9,
    1.3,
    1,
    1.7,
    0.7,
    1.3,
    0.2,
    0.1,
    -0.2,
    1.2,
    0.9,
    0.8,
    0.6,
    1.3,
    1.6,
    4.5,
    3.1,
    2.7,
    3.1,
    0.7,
    1.1,
    1.9,
    2.5,
    0.7,
    0.7,
    # 0.8,
    # 0.7,
    # 1.5,
]

# Індекс реальної заборітної плати
# з 01.19 до 12.21
# real_salary_index = [
#     86.4,
#     101.7,
#     107.6,
#     99.2,
#     99.0,
#     105.8,
#     102.4,
#     96.3,
#     100.7,
#     99.7,
#     99.5,
#     115.1,
#     87.3,
#     101.4,
#     104.8,
#     90.4,
#     100.8,
#     109.5,
#     102.6,
#     97.2,
#     104.3,
#     100.5,
#     97.2,
#     117.2,
#     85.9,
#     100.7,
#     106.7,
#     98.8,
#     98.4,
#     105.8,
#     100.1,
#     97.8,
#     100.5,
#     97.8,
#     100.9,
#     121.5,
# ]
# з 01.19 до 12.21
# unemployment_rate = [
#     9.6,
#     9.6,
#     9.6,
#     8.8,
#     8.8,
#     8.8,
#     8.4,
#     8.4,
#     8.4,
#     8.6,
#     8.6,
#     8.6,
#     8.9,
#     8.9,
#     8.9,
#     9.6,
#     9.6,
#     9.6,
#     9.7,
#     9.7,
#     9.7,
#     9.9,
#     9.9,
#     9.9,
#     10.9,
#     10.9,
#     10.9,
#     10.3,
#     10.3,
#     10.3,
#     10,
#     10,
#     10,
#     10.3,
#     10.3,
#     10.3,
# ]


def logarithmic_transformation(arr):

    transformed_arr = np.log(arr[12:]) - np.log(arr[:-12])
    return transformed_arr


cpi = list(np.array(cpi) + 100)

ln_gdp_change = logarithmic_transformation(gdp18 + gdp)
ln_gdp_deflator_change = logarithmic_transformation(gdp18_deflator + gdp_deflator)
ln_cpi_change = logarithmic_transformation(cpi)
# ln_real_salary_index_change = logarithmic_transformation(real_salary_index)
# ln_unemployment_rate_change = logarithmic_transformation(unemployment_rate)

gdp_lag0 = gdp[12:]  # 20 - 22 роки
ln_gdp_change_lag0 = ln_gdp_change[12:]
gdp_deflator_lag0 = gdp_deflator[12:]
ln_gdp_deflator_change_lag0 = ln_gdp_deflator_change[12:]
cpi_lag0 = cpi[12:]
ln_cpi_change_lag0 = ln_cpi_change[12:]
# real_salary_index_lag0 = real_salary_index[12:]
# unemployment_rate_lag0 = unemployment_rate[12:]


gdp_lag3 = gdp[9:-3]
ln_gdp_change_lag3 = ln_gdp_change[9:-3]
ln_gdp_deflator_change_lag3 = ln_gdp_deflator_change[9:-3]
gdp_deflator_lag3 = gdp_deflator[9:-3]
cpi_lag3 = cpi[9:-3]
ln_cpi_change_lag3 = ln_cpi_change[9:-3]
# real_salary_index_lag3 = real_salary_index[9:-3]
# unemployment_rate_lag3 = unemployment_rate[9:-3]

# print(len(unemployment_rate_lag3))
# print(unemployment_rate_lag3[0], unemployment_rate_lag3[-1])

gdp_lag6 = gdp[6:-6]
ln_gdp_change_lag6 = ln_gdp_change[6:-6]
gdp_deflator_lag6 = gdp_deflator[6:-6]
ln_gdp_deflator_change_lag6 = ln_gdp_deflator_change[6:-6]
cpi_lag6 = cpi[6:-6]
ln_cpi_change_lag6 = ln_cpi_change[6:-6]
# real_salary_index_lag6 = real_salary_index[6:-6]
# unemployment_rate_lag6 = unemployment_rate[6:-6]

gdp_lag9 = gdp[3:-9]
ln_gdp_change_lag9 = ln_gdp_change[3:-9]
gdp_deflator_lag9 = gdp_deflator[3:-9]
ln_gdp_deflator_change_lag9 = ln_gdp_deflator_change[3:-9]
cpi_lag9 = cpi[3:-9]
ln_cpi_change_lag9 = ln_cpi_change[3:-9]
# real_salary_index_lag9 = real_salary_index[3:-9]
# unemployment_rate_lag9 = unemployment_rate[3:-9]

gdp_lag12 = gdp[:-12]
ln_gdp_change_lag12 = ln_gdp_change[:-12]
gdp_deflator_lag12 = gdp_deflator[:-12]
ln_gdp_deflator_change_lag12 = ln_gdp_deflator_change[:-12]
cpi_lag12 = cpi[:-12]
ln_cpi_change_lag12 = ln_cpi_change[:-12]
# real_salary_index_lag12 = real_salary_index[:-12]
# unemployment_rate_lag12 = unemployment_rate[:-12]


correlation1 = round(np.corrcoef(npl_fact, gdp_lag0)[0, 1], 3)
correlation2 = round(np.corrcoef(npl_fact, gdp_lag3)[0, 1], 3)
correlation3 = round(np.corrcoef(npl_fact, gdp_lag6)[0, 1], 3)
correlation4 = round(np.corrcoef(npl_fact, gdp_lag9)[0, 1], 3)
correlation5 = round(np.corrcoef(npl_fact, gdp_lag12)[0, 1], 3)
print("GDP: ")
print(correlation1, correlation2, correlation3, correlation4, correlation5)


correlation1 = round(np.corrcoef(npl_fact, ln_gdp_change_lag0)[0, 1], 3)
correlation2 = round(np.corrcoef(npl_fact, ln_gdp_change_lag3)[0, 1], 3)
correlation3 = round(np.corrcoef(npl_fact, ln_gdp_change_lag6)[0, 1], 3)
correlation4 = round(np.corrcoef(npl_fact, ln_gdp_change_lag9)[0, 1], 3)
correlation5 = round(np.corrcoef(npl_fact, ln_gdp_change_lag12)[0, 1], 3)
print("LN_GDP_change: ")
print(correlation1, correlation2, correlation3, correlation4, correlation5)

correlation1 = round(np.corrcoef(npl_fact, gdp_deflator_lag0)[0, 1], 3)
correlation2 = round(np.corrcoef(npl_fact, gdp_deflator_lag3)[0, 1], 3)
correlation3 = round(np.corrcoef(npl_fact, gdp_deflator_lag6)[0, 1], 3)
correlation4 = round(np.corrcoef(npl_fact, gdp_deflator_lag9)[0, 1], 3)
correlation5 = round(np.corrcoef(npl_fact, gdp_deflator_lag12)[0, 1], 3)
print("GDP_Deflator: ")
print(correlation1, correlation2, correlation3, correlation4, correlation5)

correlation1 = round(np.corrcoef(npl_fact, ln_gdp_deflator_change_lag0)[0, 1], 3)
correlation2 = round(np.corrcoef(npl_fact, ln_gdp_deflator_change_lag3)[0, 1], 3)
correlation3 = round(np.corrcoef(npl_fact, ln_gdp_deflator_change_lag6)[0, 1], 3)
correlation4 = round(np.corrcoef(npl_fact, ln_gdp_deflator_change_lag9)[0, 1], 3)
correlation5 = round(np.corrcoef(npl_fact, ln_gdp_deflator_change_lag12)[0, 1], 3)
print("LN_GDP_Deflator_change: ")
print(correlation1, correlation2, correlation3, correlation4, correlation5)

correlation1 = round(np.corrcoef(npl_fact, cpi_lag0)[0, 1], 3)
correlation2 = round(np.corrcoef(npl_fact, cpi_lag3)[0, 1], 3)
correlation3 = round(np.corrcoef(npl_fact, cpi_lag6)[0, 1], 3)
correlation4 = round(np.corrcoef(npl_fact, cpi_lag9)[0, 1], 3)
correlation5 = round(np.corrcoef(npl_fact, cpi_lag12)[0, 1], 3)
print("CPI: ")
print(correlation1, correlation2, correlation3, correlation4, correlation5)

correlation1 = round(np.corrcoef(npl_fact[12:], ln_cpi_change_lag0)[0, 1], 3)
correlation2 = round(np.corrcoef(npl_fact[12:], ln_cpi_change_lag3)[0, 1], 3)
correlation3 = round(np.corrcoef(npl_fact[12:], ln_cpi_change_lag6)[0, 1], 3)
correlation4 = round(np.corrcoef(npl_fact[12:], ln_cpi_change_lag9)[0, 1], 3)
correlation5 = round(np.corrcoef(npl_fact[12:], ln_cpi_change_lag12)[0, 1], 3)
print("LN_CPI_change: ")
print(correlation1, correlation2, correlation3, correlation4, correlation5)


import statsmodels.api as sm
import statsmodels.formula.api as smf

# Linear Regression

# data = pd.DataFrame(
#     {
#         "gdp_lag12": gdp_lag12,
#         "ln_gdp_change_lag6": ln_gdp_change_lag6,
#         "npl_fact": npl_fact,
#     }
# )

# X = sm.add_constant(data[["gdp_lag12", "ln_gdp_change_lag6"]])
# y = data["npl_fact"]

# model = sm.OLS(y, X).fit()

# params = model.params
# bse = model.bse
# tvalues = model.tvalues
# pvalues = model.pvalues
# rsquared = model.rsquared


# print("\nLinear Regression\n")
# print("\nCoefficient, Std. Error, t-value, p-value")
# for i in range(len(params)):
#     print(f"b{i}: {params[i]}, {bse[i]}, {tvalues[i]}, {pvalues[i]}")
# print(f"\nR^2: {rsquared}")

# Beta Regression

import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.DataFrame(
    {
        "gdp_lag12": gdp_lag12,
        "ln_gdp_change_lag6": ln_gdp_change_lag6,
        "npl_fact": npl_fact,
    }
)

X = sm.add_constant(data[["gdp_lag12", "ln_gdp_change_lag6"]])
y = data["npl_fact"]


y_transformed = np.log(y / (1 - y))

model = sm.OLS(y_transformed, X).fit()

params = model.params
bse = model.bse
tvalues = model.tvalues
pvalues = model.pvalues
rsquared = model.rsquared

print("\nBeta Regression (logit)\n")
print("\n\nCoefficient, Std. Error, t-value, p-value")
for i in range(len(params)):
    print(f"b{i}: {params[i]}, {bse[i]}, {tvalues[i]}, {pvalues[i]}")
print(f"\nR^2: {rsquared}")

data = pd.DataFrame(
    {
        "gdp_lag12": gdp_lag12,
        "ln_gdp_change_lag6": ln_gdp_change_lag6,
        "npl_fact": npl_fact,
    }
)

X = sm.add_constant(data[["gdp_lag12", "ln_gdp_change_lag6"]])
y = data["npl_fact"]


y_transformed = np.log(y)

model = sm.OLS(y_transformed, X).fit()

params = model.params
bse = model.bse
tvalues = model.tvalues
pvalues = model.pvalues
rsquared = model.rsquared

print("\nBeta Regression (log)\n")
print("\n\nCoefficient, Std. Error, t-value, p-value")
for i in range(len(params)):
    print(f"b{i}: {params[i]}, {bse[i]}, {tvalues[i]}, {pvalues[i]}")
print(f"\nR^2: {rsquared}")

# Predicting


npl_new = [
    0.381,
    0.382,
    0.385,
    0.388,
    0.393,
    0.391,
    0.389,
    0.393,
    0.385,
    0.379,
    0.377,
    0.37,
]


gdp23 = [
    454179000000,
    454179000000,
    454179000000,
    487980333333,
    487980333333,
    487980333333,
    # 592785333333,
    # 592785333333,
    # 592785333333,
    # 644330333333,
    # 644330333333,
    # 644330333333,
]

gdp_lag12_new = gdp[-18:]
ln_gdp_change_lag6_new = logarithmic_transformation(gdp[-24:] + gdp23)

new_data = pd.DataFrame(
    {"gdp_lag12": gdp_lag12_new, "ln_gdp_change_lag6": ln_gdp_change_lag6_new}
)

X_new = sm.add_constant(new_data)

logit_predictions = model.predict(X_new)

predictions = np.exp(logit_predictions) / (1 + np.exp(logit_predictions)) + 0.1

print("Predicted Probabilities:", predictions)


months = [
    "07.22",
    "08.22",
    "09.22",
    "10.22",
    "11.22",
    "12.22",
    "01.23",
    "02.23",
    "03.23",
    "04.23",
    "05.23",
    "06.23",
    "07.23",
    "08.23",
    "09.23",
    "10.23",
    "11.23",
    "12.23",
]

plt.figure(figsize=(13, 6))
plt.plot((npl_fact[-6:] + npl_new), label="Реальні дані")
plt.plot(predictions, label="Прогноз")
plt.xlabel("Місяць")
plt.ylabel("NPL")
plt.title("Частка непрацюючих кредитiв")
plt.legend()
plt.grid(True)

plt.xticks(range(len(months)), months)
plt.ylim(0, 1)
plt.show()
