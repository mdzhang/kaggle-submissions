import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

df = pd.read_csv('train.csv')

#  df.iloc[:, :5].head()
#  df.iloc[:, 5:].head()

cat_cols = ['Pclass', 'Sex', 'Embarked']
df[cat_cols] = df[cat_cols].astype('category')

num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
# other: Cabin, Name, PassengerId, Ticket

#  plt.cla()
#  plt.clf()
#  plt.close()

fig, axes = plt.subplots(nrows=2, ncols=len(cat_cols + num_cols))

plt_conf = [
    *[{
        'name': c,
        'plt_type': 'catplot',
    } for c in cat_cols],
    *[{
        'name': c,
        'plt_type': 'regplot'
    } for c in num_cols],
]

for idx, col in enumerate(plt_conf):
    kwargs = dict(x=col['name'], y='Survived', data=df)
    if col['plt_type'] == 'catplot':
        fn = sns.catplot
        kwargs['kind'] = 'bar'
        kwargs['ax'] = axes[0][idx]
    else:
        fn = sns.regplot
        kwargs['logistic'] = True
        kwargs['ax'] = axes[1][idx - len(cat_cols)]

    fn(**kwargs)

plt.show()

notes = """
From the above visualization, we can tell:

that there is
    a strong positive correlation between survival and:
        lower Pclass
        higher fare

    a positive correlation between survival and:
        Parch

    a negative correlation between survival and:
        age
        SibSp

that women were much more likely to survive

that point of embarkment and surivval are related w/
    C being most likely to survive
    followed by Q and lastly S
"""
