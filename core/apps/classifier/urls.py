from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('review/', views.review, name='review'),
    path('predict/', views.predict, name='predict')

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)