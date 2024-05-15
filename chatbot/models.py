from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse


class Message(models.Model):
    text = models.TextField()
    category = models.CharField(max_length=100)


class Feedback(models.Model):
    user = models.CharField(max_length=255, verbose_name='Ползователь')
    email = models.EmailField(null=True)
    subject = models.TextField(verbose_name='Сообщение')

    def __str__(self):
        return f'{self.email}'


class News(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    img = models.ImageField(upload_to='photos/')
    created_at = models.DateField(auto_now_add=True)

    def get_absolute_url(self):
        return reverse('news_detail', kwargs={'pk': self.pk})

    def __str__(self):
        return self.title


class Subscribes(models.Model):
    email = models.EmailField()

    def __str__(self):
        return self.email