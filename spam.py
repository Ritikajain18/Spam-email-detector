import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('spam.csv')
df.head()
df.groupby('Category').describe()

df['spam'] = df['Category'].apply(lambda x:1 if x =='spam' else 0)
df

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, train_size = 0.8)

v  =CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:3]

model = MultinomialNB()
model.fit(X_train_count, y_train)

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
model.predict(emails_count)

X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)

clf = Pipeline([
    ('vectorizer', CountVectorizer()),('nb',MultinomialNB())
])

clf.fit(X_train, y_train)
clf.score(X_test, y_test)
clf.predict(emails)