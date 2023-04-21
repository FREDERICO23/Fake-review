import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import re
import os
from django.conf import settings

le_path = os.path.join(settings.BASE_DIR, 'core', 'apps', 'classifier', 'labelencoder.pkl')
rfc_path = os.path.join(settings.BASE_DIR, 'core', 'apps', 'classifier', 'rfc_model.pkl')
cv_path = os.path.join(settings.BASE_DIR, 'core', 'apps', 'classifier', 'vectorizer.pkl')

"""Load the saved model & vectorizer"""
le = pickle.load(open(le_path, 'rb'))
rfc = pickle.load(open(rfc_path, 'rb'))
cv = pickle.load(open(cv_path, 'rb')) 

""" Initialize Tokenizer Stemmer and Vectorizer """
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
stemmer = SnowballStemmer("english")


def home(request):
    return render(request, 'home.html')

def review(request):
    return render(request, 'predict.html')


def predict(request):
    """READ url from the website form and predict"""
    if request.method == 'POST':
        to_predict_list = request.POST['review']
        
        review_url=[to_predict_list] 

        url = cv.transform(review_url).toarray() # convert text to bag of words model (Vector)
        phish = rfc.predict(url) # predict Whether the url is good or bad
        phish = le.inverse_transform(phish) # find the url corresponding with the predicted value
        val = phish[0]
        
        return render(request, 'predict.html', {'prediction': 'The review is {}'.format(val)})
    else:
        return HttpResponse("Invalid request method")
