from distutils.log import debug
#from urllib import request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('omw-1.4')
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
wordnet_lemmatizer = WordNetLemmatizer()
from flask import Flask,render_template,request

app=Flask(__name__)
df1 = pd.read_csv('Tweets.csv')
#print(df)

# u=t.get()
# print(u)
@app.route('/')
def home():
    return render_template('index.html')
def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ", tweet)
    only_letters = only_letters.lower()
    only_letters = only_letters.split()
    filtered_result = [word for word in only_letters if word not in stopwords.words('english')]
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    lemmas = ' '.join(lemmas)
    return lemmas    
@app.route('/classify',methods=['POST'])  
def test():
    df = shuffle(df1)
    y = df['airline_sentiment']
    x = df.text.apply(normalizer)

    vectorizer = CountVectorizer()
    x_vectorized = vectorizer.fit_transform(x)

    train_x,val_x,train_y,val_y = train_test_split(x_vectorized,y)

    regressor = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    print('process finish')
    model = regressor.fit(train_x, train_y)

    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
    gs_clf = GridSearchCV(model, params, n_jobs=1, cv=5)
    gs_clf = gs_clf.fit(train_x, train_y)
    model = gs_clf.best_estimator_

    y_pred = model.predict(val_x)

    _f1 = f1_score(val_y, y_pred, average='micro')
    _confusion = confusion_matrix(val_y, y_pred)
    __precision = precision_score(val_y, y_pred, average='micro')
    _recall = recall_score(val_y, y_pred, average='micro')
    _statistics = {'f1_score': _f1,
                 'confusion_matrix': _confusion,
                 'precision': __precision,
                 'recall': _recall
                 }
    rev=request.form['review']
    print(type(rev),rev)
    test_feature = vectorizer.transform([rev])
    out=model.predict(test_feature)
    out1=out[0]
    return render_template('base.html',pre=out1)

   
if __name__=='__main__':
    app.run(debug=True)