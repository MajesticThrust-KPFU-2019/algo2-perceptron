"""

"""

import pandas as pd

# dff_test = pd.read_excel("dff_test.xlsx")
dff_test = pd.read_csv("dff_test.csv", sep=",")
vk_perc = pd.read_csv("vk_perc.csv", sep=";")

print("dff_test")
print(dff_test.head(10))

print("vk_perc")
print(vk_perc.head(10))
