from django.http import HttpResponse
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import os
from django.conf import settings
from django.contrib.auth import get_user_model
from django.shortcuts import render, redirect
from django.contrib.auth import login,logout, authenticate
from django.contrib import messages
from .forms import SignUpForm, UserLoginForm

User = get_user_model()

def register(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful." )
            return redirect("home")
        messages.error(request, "Unsuccessful registration. Invalid information.")
    form = SignUpForm()
    return render (request,"signup.html", {"register_form":form})

def logout_request(request):
    logout(request)
    messages.info(request, "You've successfully logged out")
    return redirect("home")

def login_request(request):
    if request.method == "POST":
        form = UserLoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}.")
                return redirect("review")
            else:
                messages.error(request,"Invalid username or password.")
        else:
            messages.error(request,"Invalid username or password.")
    form = UserLoginForm()
    return render(request, "login.html", {"login_form":form})



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
