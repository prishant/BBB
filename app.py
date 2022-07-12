from flask import Flask, render_template, url_for, request
from keras.models import load_model
import sys
import os
import glob
import re
import numpy as np
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

app=Flask(__name__)
MODEL_PATH = 'my_model.h5'

# Load the trained model
model = load_model(MODEL_PATH)

@app.route("/home")
@app.route("/")
def home():
    return render_template("index.html")

def preprocess(csvpath):
    train=pd.read_csv(csvpath)
    train['combine'] = train.values.tolist()
    for i in range(train['combine'].shape[0]):
            train['combine'][i]=np.asarray(train['combine'][i])
    data = []
    for i in range(0,train.shape[0]):
                data.append(train['combine'][i])
    
    new_train=np.asarray(data) 

    new_train=new_train.reshape(new_train.shape[0],new_train.shape[1],1)
    
    return new_train

@app.route('/', methods=['POST'])
def upload():
        # Get the file from post request
        f = request.files['imgfile']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))

        f.save(file_path)


        y_pred= model.predict(preprocess(file_path))

        df=pd.read_csv('Books_100k.csv')
        emotion2genre_matrix=np.array([[10,13,29,6,17,21,28,6,10,28,9,15,8,11,10,17,8,16,14,23,23],                                  #disgust
                                       [6,12,54,4,6,28,30,8,8,52,11,16,9,14,11,15,11,38,13,14,14],                                   #fear
                                       [8,10,35,4,11,21,23,7,8,38,10,13,8,23,20,23,9,29,12,15,21],                                   #sad
                                       [34,25,35,9,40,40,50,24,27,33,13,37,18,19,13,22,10,40,40,51,56],                              #neutral
                                       [24,16,33,5,30,38,56,19,17,34,13,44,11,12,13,17,6,45,41,52,57]]                              #happy 
                                       )

        emotion2genre_matrix = normalize(emotion2genre_matrix, axis=1, norm='l2')
        y_pred=np.matmul(y_pred,emotion2genre_matrix)
        y_pred=normalize(y_pred, axis=1, norm='l2')
        y_pred=y_pred[0]

        ones = np.ones((1055, 21))
        y_pred=y_pred*ones


        genres=["Business", "Classics", "Comics", "Contemporary", "Crime", "Fantasy", "Fiction",
        "History", "Horror", "Humor", "Manga", "Mystery", "Nonfiction","Philosophy", "Poetry",
        "Psychology", "Religion", "Romance", "Science", "Suspense","Thriller"]
        vectors = df[genres].values
        normalized_vectors = normalize(vectors, axis=1, norm='l2')
        similarity = cosine_similarity(y_pred, normalized_vectors)
        
        df['similarity'] = similarity[0]



        books= df.sort_values(by=['similarity', 'rating'], ascending=False) \
            .head(20)['title'] \
            .sample(frac=0.5)
        book_1=books.iloc[0]
        book_2=books.iloc[1]
        book_3=books.iloc[2]
        book_4=books.iloc[3]
        book_5=books.iloc[4]



        return render_template('index.html', book_1=book_1, book_2=book_2,
                           book_3=book_3, book_4=book_4, book_5=book_5)

if __name__ =="__main__":
    app.run(debug=True)