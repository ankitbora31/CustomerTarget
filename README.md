# CustomerTarget
Targetting a individual group of customers for the sale purpose
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')
print(df.info())

df = df.rename(columns={
    'Annual Income (k$)':'Annual_Income',
    'Spending Score (1-100)':'Spending_Score'
})
print(df.head())


gender = df['Gender'].value_counts()
labels = ['Female', 'Male']
size = df['Gender'].value_counts()
colors = ['lightgreen', 'orange']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, colors=colors, explode=explode, labels=labels, shadow=True, autopct='%.2f%%')
plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
print(gender)

age18_25 = df.Age[(df.Age <= 25) & (df.Age >= 18)]
age26_35 = df.Age[(df.Age <= 35) & (df.Age >= 26)]
age36_45 = df.Age[(df.Age <= 45) & (df.Age >= 36)]
age46_55 = df.Age[(df.Age <= 55) & (df.Age >= 46)]
age55above = df.Age[df.Age >= 56]

X = ["18-25", "26-35", "36-45", "46-55", "55+"]
y = [len(age18_25.values), len(age26_35.values),
     len(age36_45.values), len(age46_55.values),
     len(age55above.values)]

plt.figure(figsize=(15, 6))
sns.barplot(x=X, y=y, palette="rocket")
plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()

spending = df.Spending_Score
score1_20 = spending[(spending >= 1) & (spending <= 20)]
score21_40 = spending[(spending >= 21) & (spending <= 40)]
score41_60 = spending[(spending >= 41) & (spending <= 60)]
score61_80 = spending[(spending >= 61) & (spending <= 80)]
score81_100 = spending[(spending >= 81) & (spending <= 100)]

scorex = ["1-20", "21-40", "41-60", "61-80", "81-100"]
scorey = [len(score1_20.values), len(score21_40.values),
          len(score41_60.values), len(score61_80.values),
          len(score81_100.values)]

plt.figure(figsize=(15,10))
sns.barplot(x=scorex, y=scorey, palette="nipy_spectral_r")
plt.title("Spending Scores")
plt.xlabel("Score")
plt.ylabel("Number of Customer Having the Score")
plt.show()


income = df.Annual_Income
annual0_30 = income[(income >= 0) & (income <= 30)]
annual31_60 = income[(income >= 31) & (income <= 60)]
annual61_90 = income[(income >= 61) & (income <= 90)]
annual91_120 = income[(income >= 91) & (income <= 120)]
annual121_150 = income[(income >= 121) & (income <= 150)]

annualx = ["$0 - 30,000", "$30,001 - 60,000", "$60,001 - 90,000", "$90,001 - 120,000", "$120,001 - 150,000"]
annualy = [len(annual0_30.values), len(annual31_60.values),
       len(annual61_90.values), len(annual91_120.values),
       len(annual121_150.values)]

plt.figure(figsize=(15,10))
sns.barplot(x=annualx, y=annualy, palette="Set2")
plt.title("Annual Incomes")
plt.xlabel("Income")
plt.ylabel("Number of Customer")
plt.show()


x = df.iloc[:, [3, 4]].values
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'lowly')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'normal')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'red', label = 'target')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'careless')
plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = 'careful')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('K Means Clustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}


        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)
df.head()
# male = 0
# female = 1


new = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(df.iloc[:, 1:])
    new.append(kmeans.inertia_)
plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), new, linewidth=2, color="red", marker="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1, 11, 1))
plt.ylabel("WCSS")
plt.show()

km = KMeans(n_clusters=5)
clusters = km.fit_predict(df.iloc[:, 1:])
df["label"] = clusters

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual_Income"][df.label == 0], df["Spending_Score"][df.label == 0], c='blue',
           s=60)
ax.scatter(df.Age[df.label == 1], df["Annual_Income"][df.label == 1], df["Spending_Score"][df.label == 1], c='red',
           s=60)
ax.scatter(df.Age[df.label == 2], df["Annual_Income"][df.label == 2], df["Spending_Score"][df.label == 2], c='green',
           s=60)
ax.scatter(df.Age[df.label == 3], df["Annual_Income"][df.label == 3], df["Spending_Score"][df.label == 3], c='orange',
           s=60)
ax.scatter(df.Age[df.label == 4], df["Annual_Income"][df.label == 4], df["Spending_Score"][df.label == 4], c='purple',
           s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()
