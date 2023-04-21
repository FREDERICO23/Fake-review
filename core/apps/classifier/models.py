from django.db import models

# Create your models here.
class ClassifiedReviews(models.Model):
    review = models.CharField(max_length=500)
    results = models.CharField(max_length=50)