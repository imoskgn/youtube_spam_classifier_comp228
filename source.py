import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import string
import os


# 1. Load the data
path = "../"
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

### Converting tfidf to pandas DF to shuffle and split it
df_train_tfidf = pd.DataFrame()
df_train_tfidf = pd.concat([df_train_tfidf, pd.DataFrame(train_tfidf.todense())], axis=1)
df_train_tfidf.columns = count_vectorizer.get_feature_names_out()

train_data = pd.DataFrame()
test_data = pd.DataFrame()
# Concatenating data + classification, last column is the Classification
concat_data = pd.concat([df_train_tfidf, data.CLASS], axis=1)
# Shuffling the data
concat_data = concat_data.sample(frac=1)
## Splitting data
train_data = concat_data[0:263]
test_data = concat_data[263::]

# 8. Fitting the model
classifier = MultinomialNB()
classifier.fit(train_data.iloc[:, 0:1219], train_data.iloc[:, 1219])

# 9. Crossvalidate
folds = 5
scores = accuracy_values = cross_val_score(
    classifier,
    concat_data.iloc[:, 0:1219],
    concat_data.iloc[:, 1219],
    scoring='accuracy',
    cv=folds
)
print(scores.mean())

# 10. Testing the model
predictions = classifier.predict(test_data.iloc[:, 0:1219])

# Confussion Matrix
conf_matrix = confusion_matrix(predictions, test_data.iloc[:, 1219])
print("Confusion Matrix: ")
print(conf_matrix)

# Accuracy score
print("Accuracy Score: " + str(accuracy_score(test_data.iloc[:, 1219], predictions)))
print(classification_report(test_data.iloc[:, 1219], predictions))

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

predictions = classifier.predict(input_tfidf)
