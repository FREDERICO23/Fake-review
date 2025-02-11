from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from django.utils.translation import gettext_lazy as _
from django.utils import timezone

from .managers import CustomUserManager

class User(AbstractBaseUser, PermissionsMixin):
    # county choices
    username = models.CharField(max_length=30, unique=True)
    email = models.EmailField(_('email address'), unique=True)
    name =  models.CharField(max_length=300, blank=True)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    date_joined = models.DateTimeField(default=timezone.now)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username','name']
    
    objects = CustomUserManager()

    def __str__(self):
        return self.username

class ClassifiedReviews(models.Model):
    review = models.CharField(max_length=500)
    result = models.CharField(max_length=50)
    date = models.DateTimeField(default=timezone.now)