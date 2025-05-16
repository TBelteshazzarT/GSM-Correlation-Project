import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

iris_data = pd.read_csv("Iris.csv")

print(iris_data.head())
print(iris_data.shape)
print(iris_data.value_counts("Species"))

#sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris_data)
#sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=iris_data)
#sns.pairplot(iris_data.drop(['Id'], axis=1), hue='Species', height=2)
#plt.show()

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,89,87,88,111,89,103,87,94,78,77,85,86]
slope, intercept, r, p, std_err = stats.linregress(x,y)

def myfunc(x):
    return slope * x +intercept

mymodel = list(map(myfunc, x))
print(r)
print(myfunc(10))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
