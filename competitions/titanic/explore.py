import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

df = pd.read_csv('train.csv')

CABIN_RE = re.compile(r'([A-Z]{1})(\d+)')

# cabin codes and numbers generally are ordered and indicate
# location aboard the ship; split into categorical CabinCode
# and continuous CabinNumber

EMPTY_CABIN_SER = pd.Series({
    'CabinCode': np.nan,
    'CabinNumber': np.nan,
})


def extract_code(val):
    if val is None:
        return EMPTY_CABIN_SER
    # TODO: maybe account for multiple cabins, and average
    #       cabin code e.g. 'C23 C25 C27' => ('C', 25)
    cabin = str(val).split(' ')[-1]
    m = re.match(CABIN_RE, cabin)
    if not m:
        return EMPTY_CABIN_SER
    code, num = m.group(1), m.group(2)
    return pd.Series({'CabinCode': code, 'CabinNumber': num})


df[['CabinCode', 'CabinNumber']] = df['Cabin'].apply(extract_code)

#  df.iloc[:, :5].head()
#  df.iloc[:, 5:].head()

# TODO: PassengerId is probably useless, but I haven't confirmed
# TODO: If name indicates ethnic background, could be a predictor
# Cabin not useful after extraction
df = df.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
cat_cols = ['Pclass', 'Sex', 'Embarked', 'CabinCode']
df[cat_cols] = df[cat_cols].astype('category')

num_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'CabinNumber']
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

df2 = df[['Survived', 'CabinNumber']].dropna()
df2['CabinNumber'] = df2['CabinNumber'].astype('int64')
sns.regplot(x='CabinNumber', y='Survived', data=df2, logistic=True)

df3 = df[['Survived', 'CabinCode']].dropna()
df3['CabinCode'] = df3['CabinCode'].astype('category')
sns.catplot(x='CabinCode', y='Survived', data=df3, kind='bar')

# TODO
df2 = df[['Survived', 'CabinNumber', 'CabinCode']].dropna()
df2['CabinNumber'] = df2['CabinNumber'].astype('int64')
df2['CabinCode'] = df2['CabinCode'].astype('category')
sns.regplot(
    x='CabinNumber', y='Survived', hue='CabinCode', data=df2, logistic=True)
