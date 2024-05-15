from modeltranslation.translator import translator, TranslationOptions
from .models import *


class NewsTranslationOptions(TranslationOptions):
    fields = ('title', 'content',)


translator.register(News, NewsTranslationOptions)
