from captcha.fields import CaptchaField
from django import forms
from django.utils.translation import gettext_lazy as _
from .models import *
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User


class FeedbackForm(forms.ModelForm):
    user = forms.CharField(widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Adyňyz'}))
    email = forms.EmailField(widget=forms.EmailInput(
        attrs={'class': 'form-control', 'placeholder': 'Email'}))
    subject = forms.CharField(widget=forms.Textarea(
        attrs={'class': 'form-control', 'placeholder': 'Hat'}))
    captcha = CaptchaField()

    class Meta:
        model = Feedback
        fields = ['user', 'email', 'subject']


class SignupForm(UserCreationForm):
    username = forms.CharField(widget=forms.TextInput(
        attrs={'placeholder': 'Adyňyz'}))
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'placeholder': "Email"}))
    password1 = forms.CharField(widget=forms.PasswordInput(
        attrs={'placeholder': 'Parol'}))
    password2 = forms.CharField(widget=forms.PasswordInput(
        attrs={'placeholder': 'Paroly tassykla'}))

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')


class UserLoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(
        attrs={'placeholder': 'Ulanyjy adyňyz'}))
    password = forms.CharField(widget=forms.PasswordInput(
        attrs={'placeholder': 'Parolyňyz'}))


class SubscribeForm(forms.ModelForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={
        'class': 'form-control border-0 rounded-pill w-100 ps-4 pe-5',
        'style': 'height: 48px;',
        'placeholder': 'Email',
    }))

    class Meta:
        model = Subscribes
        fields = ('email', )


class NewsletterForm(forms.Form):
    subject = forms.CharField(max_length=200, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': "Tema"}))
    message = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control', 'placeholder': "Hat"}))
