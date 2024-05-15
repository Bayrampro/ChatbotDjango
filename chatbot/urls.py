from django.urls import path

import chatbot.views
from .views import *

urlpatterns = [
    path('', home, name='home'),
    path('chat/', chat, name='chat'),
    # path('get_response/', get_response, name='get_response'),

    # here is confirmation urls
    path('register/', signup, name='register'),
    path('activate/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/',
         activate, name='activate'),
    path('confirm/', confirm, name='confirm'),
    path('success/', success, name='success'),
    path('logout/', signout, name='logout'),
    path('login/', user_login, name='login'),

    # here is reset password urls
    path('reset_password/', CustomPasswordResetView.as_view(), name='password_reset'),
    path('reset_password_sent/', CustomPasswordResetDoneView.as_view(), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', CustomPasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('reset_password_complete/', CustomPasswordResetCompleteView.as_view(), name='password_reset_complete'),

    #News
    path('news/', news, name="news"),
    path('news_detail/<int:pk>/', news_detail, name="news_detail"),

    #Habarlaşmak
    path('contact/', about, name="contact"),

    #Newsletter
    path("send_newsletter/", send_newsletter, name="send_newsletter"),
    path('predict_address/', predict_address, name='predict_address'),

    #session usti bilen fayllara chatlary sohranit etmek
    path('new/', start_new_chat, name='start_new_chat'),

    # Для страницы с диалогами пользователя
    # path('user_dialogues/', user_dialogues, name='user_dialogues'),
    # Для страницы с конкретным диалогом
    path('dialogue/<str:user>/<str:session_key>/', dialogue_detail, name='dialogue_detail'),

    path('save-to-file/', save_to_file, name='save_to_file'),

    path('delete_dialogue/<str:session_key>/', delete_dialogue, name='delete_dialogue'),
]
