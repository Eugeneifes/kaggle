import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv('train.csv', usecols=["Sex", "Survived"])

#print(train["Sex"].tolist())
#print(type(train["Sex"].tolist()))

print(train["Survived"].tolist().count(0))
print(train["Survived"].tolist().count(1))

plt.hist(train["Survived"].tolist(), bins=[0, 1, 2], rwidth=0.7)
plt.show()


"""
for i in range(len(train["Survived"])):
    print("Survived: " + str(train["Survived"][i]) + ", " + "Sex: " + str(train["Sex"][i]))
"""