from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    phone = forms.CharField(max_length=15, required=True)
    age = forms.IntegerField(min_value=18, required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'phone', 'age', 'password1', 'password2']

class UploadImageForm(forms.Form):
    image = forms.ImageField()
