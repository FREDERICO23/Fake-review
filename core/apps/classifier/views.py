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
from .models import ClassifiedReviews
from django.template.loader import get_template
from xhtml2pdf import pisa
from django.utils import timezone

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
        review = request.POST['review']
        
        review=[review] 

        review_array = cv.transform(review).toarray() # convert text to bag of words model (Vector)
        pred = rfc.predict(review_array) # predict Whether the url is good or bad
        pred = le.inverse_transform(pred) # find the url corresponding with the predicted value
        val = pred[0]
        # Save the review and its classification to the database
        ClassifiedReviews.objects.create(review=review, result=val, date=timezone.now())

        return render(request, 'predict.html', {'prediction': 'The review is {}'.format(val)})
    else:
        return HttpResponse("Invalid request method")
    
def generate_report(request):
    if not request.user.is_authenticated:
        return redirect('login')
    
    reviews = ClassifiedReviews.objects.all()

    # Generate the PDF report
    template_path = 'report.html'
    context = {'reviews': reviews}
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="report.pdf"'
    template = get_template(template_path)
    html = template.render(context)
    pisa_status = pisa.CreatePDF(html, dest=response)
    if pisa_status.err:
        return HttpResponse('An error occurred while generating the PDF.')
    return response

