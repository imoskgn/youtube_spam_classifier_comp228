import pandas
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import string

import os
# 1. Load the data
path = "/Users/oskgn/Downloads"
filename = 'Youtube01-Psy.csv'
fullpath = os.path.join(path, filename)
data = pd.read_csv(fullpath)

# 2. Data exploration
print("Head: \n", data.head(3))
print("\nData types: \n", data.dtypes)
print("\nShape: ", data.shape)
# 2. Deleting columns
data.drop(columns=['COMMENT_ID', 'AUTHOR', 'DATE'], inplace=True)

# 3. Preparing dataset
# 3.1 Deleting punctuation
stop = stopwords.words('english')
punctuation = string.punctuation
data['CONTENT'] = data['CONTENT'].apply(lambda x: ''.join([char for char in x if char not in punctuation]))
data['CONTENT'] = data['CONTENT'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
# 3. Deleting punctuation

print("Head: \n", data.head(3))
print("\nData types: \n", data.dtypes)
print("\nShape: ", data.shape)

# 3. Building a Category text predictor
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(data.CONTENT)
print(type(train_tc))
print("\nDimensions of training data:", train_tc.shape)

# 5.Downscaling the dataset
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
print("Tf-Idf Shape= ", train_tfidf.shape)

### Trying Pandas Sample
df = pandas.DataFrame()
X = train_tfidf
X1 = pd.DataFrame(X.todense(), columns=count_vectorizer.get_feature_names_out())
df = pd.concat([df, X1], axis=1)

classifier = MultinomialNB().fit(X1, data.CLASS)

input_data = pd.Series([
    'nice video',
    'please visit my website',
    'well done Tom, I wish I could do it too',
    'you have WON a PRIZE, click to see',
    'Hello Jimmy, you explained everything very well'])

# Transform input data using count vectorizer
input_tc = count_vectorizer.transform(input_data)
type(input_tc)
print(input_tc)
# Transform vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)
type(input_tfidf)
print(input_tfidf)

df1 = pandas.DataFrame()
X2 = input_tfidf
X3 = pd.DataFrame(X2.todense())
df1 = pd.concat([df1, X3], axis=1)

predictions = classifier.predict(input_tfidf)