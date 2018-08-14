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
    """Extract cabin code and number e.g. B26 => (B, 26)"""
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


def extract_name(val):
    pat = re.compile('^(.+?), (.+?)\. (.+)\s*$')
    m = re.match(pat, val)
    title = m.group(2)

    # default matches format <last name>, <title>. <first name>
    last_name = m.group(1)
    # TODO: might catch nicknames
    first_name = m.group(3)

    if title == 'Mrs':
        rest = m.group(3)

        # <husband name>(<wife name><ending>
        # where ending is ')' or '...'
        pat2 = re.compile('^(.*)\s*\((.+)(\)|\.\.\.)\s*$')
        m2 = re.match(pat2, rest)
        if m2:
            # TODO: might be truncated if ends with ...
            name = m2.group(2)
            fragments = name.split()
            first_name = fragments[0]
            last_name = fragments[-1]

    # remove nicknames e.g. Florence "Fannie"
    pat3 = re.compile('^(.+?)\s*?(".+")?$')
    m3 = re.match(pat3, first_name)
    if m3:
        first_name = m3.group(1)
    return (title, first_name, last_name)


def get_name_df(df):
    def extract_fragments(val):
        title, first_name, last_name = extract_name(val)
        return pd.Series({
            'FirstName': first_name,
            'LastName': last_name,
            'Title': title,
        })

    return df['Name'].apply(extract_fragments)

df[['FirstName', 'LastName', 'Title']] = get_name_df(df)

def get_unique_titles(df):
    """Get unique titles in names e.g. Mr, Ms, the Countess, etc."""
    names = list(df['Name'])
    unique_titles = set([extract_name(n)[0] for n in names])
    return unique_titles


#  df.iloc[:, :5].head()
#  df.iloc[:, 5:].head()

# PassengerId has no particular relationship to survival
# TODO: If name indicates ethnic background, could be a predictor
# Cabin not useful after extraction
df = df.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
cat_cols = ['Pclass', 'Sex', 'Embarked', 'CabinCode']
df[cat_cols] = df[cat_cols].astype('category')

num_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'CabinNumber']
# other: Name, PassengerId, Ticket

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
        CabinNumber

    a negative correlation between survival and:
        age
        SibSp

that women were much more likely to survive

that point of embarkment and surivval are related w/
    C being most likely to survive
    followed by Q and lastly S

that cabin codes G, A, C were also less likely to survive
"""

df2 = df[['Survived', 'CabinNumber']].dropna()
df2['CabinNumber'] = df2['CabinNumber'].astype('int64')
sns.regplot(x='CabinNumber', y='Survived', data=df2, logistic=True)

df3 = df[['Survived', 'CabinCode']].dropna()
df3['CabinCode'] = df3['CabinCode'].astype('category')
sns.catplot(x='CabinCode', y='Survived', data=df3, kind='bar')

df2 = df[['Survived', 'Name']]


def plot_cabin_codes(df):
    """Plots logreg of survival against cabin number, split by
    cabin code, with each plot in a 2 x 4 grid
    """
    df2 = df[['Survived', 'CabinNumber', 'CabinCode']].dropna()
    df2['CabinNumber'] = df2['CabinNumber'].astype('int64')
    df2['CabinCode'] = df2['CabinCode'].astype('category')

    fig, axes = plt.subplots(nrows=2, ncols=4)

    for idx, cc in enumerate(sorted(df2['CabinCode'].unique())):
        df3 = df2[df2['CabinCode'] == cc]
        row = 1 if idx > 3 else 0
        col = (idx - 4) if idx > 3 else idx

        ax = sns.regplot(
            x='CabinNumber',
            y='Survived',
            data=df3,
            logistic=True,
            ax=axes[row, col])
        ax.set(xlabel=cc)
    plt.show()
