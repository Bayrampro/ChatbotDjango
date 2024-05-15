import datetime
import re

import pandas
import pytz
import requests
from django.contrib.auth.models import User
from django.utils.translation import gettext_lazy as _
from django.contrib import messages
from django.contrib.auth import login, logout, get_user_model
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.views import PasswordResetView, PasswordResetConfirmView, PasswordResetDoneView, \
    PasswordResetCompleteView
from django.contrib.sites.shortcuts import get_current_site
from django.core.paginator import Paginator
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.template.loader import render_to_string
from django.utils.encoding import force_str, force_bytes
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from nltk import WordNetLemmatizer

from chatbot_project import settings
from .forms import *
from .models import News
from .token import account_activation_token
from django.core.mail import EmailMessage, send_mail

import os

from django.shortcuts import render
from django.http import JsonResponse
from joblib import load

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from nltk.stem.snowball import SnowballStemmer


def home(request):
    news = News.objects.order_by("-created_at")[0:3]
    if request.method == 'POST':
        form = SubscribeForm(request.POST)
        if form.is_valid():
            form.save()
    else:
        form = SubscribeForm()
    return render(request, 'chatbot/index.html', {'news': news, 'form': form})


#@login_required
def chat(request):
    if request.user.is_authenticated:
        username = request.user.username
        email = request.user.email

        """
    
        !!!
        Üýtgedildi baş sahypany refresh edende şol sessiýadaky soobşenýalar görkeziler ýaly
        !!!
    
        """

        dialogue_file_path = os.path.join('dialogues', f'{request.user}_{request.session.session_key}.txt')
        if os.path.exists(dialogue_file_path):
            with open(dialogue_file_path, 'r', encoding='utf-8') as file:
                messages = file.readlines()
                formatted_messages = []

                for message in messages:
                    if message.startswith('User:'):
                        formatted_messages.append(('user', message.split(':')[1].strip()))
                    elif message.startswith('Bot:'):
                        formatted_messages.append(('bot', message.split(':')[1].strip()))
                print(formatted_messages)
        else:
            formatted_messages = []
            print(formatted_messages)

        dialogue_files = os.listdir('dialogues')
        dialogues_with_session_keys = []
        current_user = request.user.username
        for dialogue_file in dialogue_files:
            file_username = dialogue_file.split('_')[0]
            if file_username == current_user:
                session_key = dialogue_file.split('_')[1].split('.')[0]
                with open(os.path.join('dialogues', dialogue_file), 'r', encoding='utf-8') as file:
                    first_line = file.readline().strip().replace('User: ', '')
                dialogues_with_session_keys.append((first_line, session_key))

        return render(request, 'chatbot/chat.html', {'username': username, 'email': email, 'messages': formatted_messages, 'dialogues_with_session_keys': dialogues_with_session_keys})
    else:
        return redirect('login')


model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.joblib')

model = load(model_path)
vectorizer = load(vectorizer_path)


# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')  # Добавлено скачивание WordNet

import pymorphy2

# Инициализация Pymorphy2
morph = pymorphy2.MorphAnalyzer()
wordnet_lemmatizer = WordNetLemmatizer()

data = pandas.read_csv('./data/dataset.csv')

def preprocess_text(text):
    # Приведение всех слов к нижнему регистру
    tokens = word_tokenize(text)
    # Разделение текста на слова
    tokens = [token.lower() for token in tokens]

    tokens = [token for token in tokens if token not in string.punctuation]
    # Лемматизация каждого слова

    stop_words_en = set(stopwords.words('english'))

    stop_words_ru = set(stopwords.words('russian'))

    stop_words = stop_words_en.union(stop_words_ru)

    words_to_remove = ['what', 'doing', 'can']

    for word in words_to_remove:
        if word in stop_words:
            stop_words.remove(word)

    tokens = [token for token in tokens if token not in stop_words]

    tokens_ru = [morph.parse(token)[0].normal_form for token in tokens]

    tokens_en = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    # Сборка текста обратно из лемматизированных слов

    tokens = tokens_en + tokens_ru
    text = ' '.join(tokens)

    return text


#####################################################

def extract_tags(text):

    """

    Döredildi!!! Wagt we Sene üçin teglar

    """

    tags = []
    # Дата
    if re.search(r'\bдата\b|\bдату\b|\bдате\b|\bдаты\b', text, re.IGNORECASE):
        tags.append('<дата>')
    # Время
    if re.search(r'\bвремя\b|\bвремени\b|\bчасов\b|\bчасы\b|\bчас\b', text, re.IGNORECASE):
        tags.append('<время>')
    # День недели
    if re.search(r'\bдень недели\b|\bсегодня\b|\bсегоднящний день\b', text, re.IGNORECASE):
        tags.append('<день_недели>')

    # Дата (английский)
    if re.search(r'\bdate\b', text, re.IGNORECASE):
        tags.append('<дата>')
    # Время (английский)
    if re.search(r'\btime\b', text, re.IGNORECASE):
        tags.append('<время>')
    # День недели (английский)
    if re.search(r'\bday of the week\b|\bweek day\b|\bday week\b', text, re.IGNORECASE):
        tags.append('<день_недели>')

    # Дата (туркменский)
    if re.search(r'\bsene\b|\bsenäni\b|\bsenani\b', text, re.IGNORECASE):
        tags.append('<дата>')
    # Время (туркменский)
    if re.search(r'\bwagt\b|\bwagty\b|\bwagtyny\b|\bsagat\b|\bsagady\b|\bsagadyny\b', text, re.IGNORECASE):
        tags.append('<время>')
    # День недели (туркменский)
    if re.search(r'\bhepde güni\b|\bhepdäniň güni\b|\bhepdanin guni\b|\bnäçinji gün\b|\bnäçinji güni\b|\bnacinji gun\b|\bnacinji guni\b|\bhaýsy gün\b|\bhaýsy güni\b|\bhaysy gun\b|\bhaysy guni\b', text, re.IGNORECASE):
        tags.append('<день_недели>')

    return tags


from langdetect import detect


def detect_language(text):

    """

    Döredildi!!! Dili saýgarýan funksiýa

    """

    try:
        return detect(text)
    except:
        return None


def get_weekday(language):

    """

    Döredildi!!! Dile görä hepde güni barada jogap berýär

    """

    weekday = datetime.datetime.now().strftime('%A')
    if language == 'en':
        return f"Today {weekday}"
    elif language == 'ru' or language == 'uk':
        trans_weekday = ''
        if weekday == 'Monday':
            trans_weekday = 'Понедельник'
        elif weekday == 'Tuesday':
            trans_weekday = 'Вторник'
        elif weekday == 'Wednesday':
            trans_weekday = 'Среда'
        elif weekday == 'Thursday':
            trans_weekday = 'Четверг'
        elif weekday == 'Friday':
            trans_weekday = 'Пятница'
        elif weekday == 'Saturday':
            trans_weekday = 'Суббота'
        elif weekday == 'Sunday':
            trans_weekday = 'Воскресенье'
        return f"Сегодня {trans_weekday}"
    elif language == 'tk' or language == 'tr':
        trans_weekday = ''
        if weekday == 'Monday':
            trans_weekday = 'Duşenbe'
        elif weekday == 'Tuesday':
            trans_weekday = 'Sişenbe'
        elif weekday == 'Wednesday':
            trans_weekday = 'Çarşenbe'
        elif weekday == 'Thursday':
            trans_weekday = 'Penşenbe'
        elif weekday == 'Friday':
            trans_weekday = 'Anna'
        elif weekday == 'Saturday':
            trans_weekday = 'Şenbe'
        elif weekday == 'Sunday':
            trans_weekday = 'Ýekşenbe'
        return f"Şu gün {trans_weekday}"
    else:
        return f"Şu gün {weekday}"


#####################################################


def predict_address(request):
    if request.method == 'GET':
        query = request.GET.get('query', None)
        if query:
            preprocessed_query = preprocess_text(query)

            print(preprocessed_query)

            tags = extract_tags(query)

            print(tags)

            query_vectorized = vectorizer.transform([preprocessed_query])
            predicted_address = model.predict(query_vectorized)[0]
            data_values1 = ' '.join(data['question'].values)
            data_values2 = data_values1.split()

            x = any(i in preprocessed_query for i in data_values2)

            weather_keywords = ["weather", "howa", "погода"]
            forecast_keywords = ["прогноз погоды", "forecast", "howa maglumaty"]
            city_names = [
                "Mary", "mary", "Lebap", "lebap", "Ashgabat", "ashgabat", "Ahal", "ahal",
                "Balkan", "balkan", "Dashoguz", "dashoguz", "Мары", "мары", "Ашхабад",
                "ашхабад", "Лебап", "лебап", "Балкан", "балкан", "Дашогуз", "дашогуз", "Ахал", "ахал"
            ]

            has_weather_keyword = any(keyword in preprocessed_query for keyword in weather_keywords)
            has_forecast_keyword = any(forecast in preprocessed_query for forecast in forecast_keywords)
            has_city_name = any(city_name in preprocessed_query for city_name in city_names)

            if has_weather_keyword and has_city_name and not has_forecast_keyword:
                # Извлечение города из запроса
                location = extract_location(query)
                if location:
                    api_key = '7bb75bca252ee310e90e9a127ca642e2'
                    api_url = f'http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric'
                    response = requests.get(api_url)
                    if response.status_code == 200:
                        weather_data = response.json()
                        weather_description = weather_data['weather'][0]['description']
                        temperature = weather_data['main']['temp']

                        ###############!!!!!!!!!!!!!##################

                        weather_response = ''

                        cloudy = ['few clouds', 'scattered clouds', 'broken clouds', 'overcast clouds']
                        foggy = ['mist', 'fog', 'smoke', 'haze']
                        drizzle = [
                            'light intensity drizzle',
                            'drizzle',
                            'heavy intensity drizzle',
                            'light intensity drizzle rain',
                            'drizzle rain',
                            'heavy intensity drizzle rain',
                            'shower rain and drizzle',
                            'heavy shower rain and drizzle',
                            'shower drizzle',
                            'light rain',
                            'moderate rain',
                            'heavy intensity rain',
                            'very heavy rain',
                            'extreme rain',
                            'freezing rain',
                            'light intensity shower rain',
                            'shower rain',
                            'heavy intensity shower rain',
                            'ragged shower rain',
                        ]
                        snow = [
                            'light snow',
                            'snow',
                            'heavy snow',
                            'sleet',
                            'shower sleet',
                            'light rain and snow',
                            'rain and snow',
                            'light shower snow',
                            'shower snow',
                            'heavy shower snow'
                        ]

                        sand = [
                            'dust whirls',
                            'sand',
                            'dust',
                            'volcanic ash',
                        ]

                        if weather_description in cloudy:
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                                f"Облачная и пасмурная погода создает<br>" \
                                                f"уютную атмосферу для посещения кафе или уютного места для чтения. <br>" \
                                                f"Можно насладиться чашечкой горячего напитка, читать книгу или просто<br>" \
                                                f"наслаждаться спокойным временем.<br>" \
                                                f"можно выбрать легкую куртку или свитер. Они помогут защитить от прохлады, <br>" \
                                                f"но при этом не будут слишком громоздкими."

                        elif weather_description in foggy:
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                                f"Это может показаться не слишком захватывающим занятием, <br>" \
                                                f"но уборка и организация дома могут приносить удовлетворение и создавать уютную обстановку. <br>" \
                                                f"Попробуйте обновить свой интерьер или организовать свои вещи.<br>" \
                                                f"Из-за ограниченной видимости важно, чтобы вас было легко заметить. <br>" \
                                                f"Поэтому выбирайте одежду ярких цветов или с отражающими элементами, <br>" \
                                                f"которые помогут вам быть более видимым в условиях тумана.<br>" \
                                                f"Туман часто сопровождается повышенной влажностью и прохладой, поэтому выбирайте теплую одежду. <br>" \
                                                f"Пальто, длинные пальто или куртки с утеплителем будут хорошим выбором."

                        elif weather_description in drizzle:
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                                f"Устройте небольшой киномарафон с вашими любимыми фильмами или сериалами. <br>" \
                                                f"Заварите себе чашку ароматного чая и наслаждайтесь просмотром. <br>" \
                                                f"Одежда с водонепроницаемым покрытием будет хорошим выбором для защиты от дождя. <br>" \
                                                f"Выберите легкий и компактный дождевик или плащ, <br>" \
                                                f"который можно легко надеть поверх вашей основной одежды."

                        elif weather_description in snow:
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                                f"Для саморазвития и обучения можно изучать новые темы онлайн, <br>" \
                                                f"писать статьи или просто наслаждаться уединенным временем, обдумывая свои мысли и идеи.<br>" \
                                                f" Наденьте утепленную куртку или пальто с ветрозащитным материалом, чтобы защититься от холода и ветра.<br>" \
                                                f"Шапка, шарф и перчатки: Эти аксессуары помогут сохранить тепло и защитить от потери тепла через голову и руки."

                        elif weather_description in sand:
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                                f"Уютное чтение — отличный способ провести время. Выберите книгу, <br>" \
                                                f"которую давно хотели прочитать, или журнал по интересующей вас теме.<br>" \
                                                f"При погоде с пылью, песком, пылевыми вихрями или вулканическим пеплом рекомендуется надеть защитную одежду и аксессуары. \n" \
                                                f"Носите защитные очки или солнцезащитные очки, <br>" \
                                                f"широкополую шляпу или панаму для защиты глаз и лица от пыли и песка. <br>" \
                                                f"Используйте маску, бандану или повязку, чтобы закрыть рот и нос и предотвратить вдыхание пыли и мелких частиц. \n" \
                                                f"Выбирайте светлую, свободносидящую одежду с длинными рукавами и брюками, <br>" \
                                                f"чтобы полностью покрыть тело и минимизировать контакт с частицами." \

                        elif weather_description == 'clear sky':
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                            f"Если у вас есть фотоаппарат или смартфон с хорошей камерой, <br>" \
                                            f"отправляйтесь на прогулку и делайте красивые фотографии природы, <br>" \
                                            f"архитектуры или просто своих близких.<br>" \
                                            f"Носите легкую рубашку, майку или футболку из натуральных материалов, <br>" \
                                            f"таких как хлопок или лен. Это поможет вам оставаться прохладным и комфортным в течение дня." \

                        elif weather_description == 'tornado':
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                            f" Если вы находитесь в помещении, возьмите на себя плотную одежд<br> <br>" \
                                            f"которая защитит от обломков и разрушений.<br>" \
                                            f"Когда погода становится неблагоприятной из-за торнадо, ваша безопасность становится главным приоритетом. <br>" \
                                            f"Поскольку в таких ситуациях важно оставаться в безопасном месте и следить за метеорологической обстановкой, <br>" \
                                            f"заняться чем-то вряд ли получится." \

                            #####################!!!!!!!!!!!!!!!!!!!!!!!#############################

                        save_dialogue(request, request.session.session_key, query, weather_response)
                        return JsonResponse(
                            {'address': weather_response}
                        )
                    else:
                        error_response = 'Ошибка при получении прогноза погоды'
                        save_dialogue(request, request.session.session_key, query, error_response)
                        return JsonResponse({'address': error_response}, status=500)
                else:
                    default_response = "Извините, не могу определить местоположение для получения погоды."
                    save_dialogue(request, request.session.session_key, query, default_response)
                    return JsonResponse({'address': default_response})
            elif has_weather_keyword and not has_city_name:
                default_response = "В каком городе вы хотите узнать погоду?"
                save_dialogue(request, request.session.session_key, query, default_response)
                return JsonResponse({'address': default_response})

            elif has_forecast_keyword and has_city_name:
                location = extract_location(query)
                if location:
                    # Запрос на прогноз погоды к API OpenWeatherMap
                    api_key = '7bb75bca252ee310e90e9a127ca642e2'
                    api_url = f'http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}&units=metric'
                    response = requests.get(api_url)

                    if response.status_code == 200:
                        weather_data = response.json()
                        formatted_forecast = format_weather_forecast(weather_data)
                        forecast_message = format_forecast_message(formatted_forecast)

                        save_dialogue(request, request.session.session_key, query, forecast_message)
                        return JsonResponse({'address': forecast_message})

                    else:
                        error_response = 'Ошибка при получении прогноза погоды'
                        save_dialogue(request, request.session.session_key, query, error_response)
                        return JsonResponse({'address': error_response}, status=500)

                else:
                    default_response = "Извините, не могу определить местоположение для получения погоды."
                    save_dialogue(request, request.session.session_key, query, default_response)
                    return JsonResponse({'address': default_response})

            elif has_weather_keyword and not has_city_name:
                default_response = "В каком городе вы хотите узнать прогноз погоды?"
                save_dialogue(request, request.session.session_key, query, default_response)
                return JsonResponse({'address': default_response})

             ################################################

            elif tags and not x:

                if '<дата>' in tags:
                    current_date = datetime.date.today()
                    date = f"Сегодняшняя дата {current_date}"
                    save_dialogue(request, request.session.session_key, query, date)
                    return JsonResponse({'address': date})
                elif '<время>' in tags:
                    tz_ashgabat = pytz.timezone('Asia/Ashgabat')
                    current_time = datetime.datetime.now(tz=tz_ashgabat).strftime('%H-%M')
                    time = f"Текущее время {current_time}"
                    save_dialogue(request, request.session.session_key, query, time)
                    return JsonResponse({'address': time})
                elif '<день_недели>' in tags:
                    lang = detect_language(query)
                    save_dialogue(request, request.session.session_key, query, get_weekday(lang))
                    return JsonResponse({'address': get_weekday(lang)})

            #################################################

            else:
                if x:
                    save_dialogue(request, request.session.session_key, query, predicted_address)
                    return JsonResponse({'address': predicted_address})
                else:
                    default_response = "Извините, не могу найти ответ на ваш вопрос."
                    save_dialogue(request, request.session.session_key, query, default_response)
                    return JsonResponse({'address': default_response})
        else:
            return JsonResponse({'error': 'Параметр query отсутствует'}, status=400)


import datetime


def format_weather_forecast(weather_data):

    """

    Döredildi!!! Forecast shablonda gowy cykar yaly!!!

    """

    formatted_forecast = {}
    forecast_items = weather_data['list']

    for item in forecast_items:
        dt_txt = item['dt_txt']
        date = dt_txt.split(' ')[0]
        time = dt_txt.split(' ')[1]
        description = item['weather'][0]['description']
        temperature = item['main']['temp']

        if date not in formatted_forecast:
            formatted_forecast[date] = {'descriptions': [], 'temperatures': []}

        formatted_forecast[date]['descriptions'].append(description)
        formatted_forecast[date]['temperatures'].append(temperature)

    return formatted_forecast


def format_forecast_message(formatted_forecast):

    """

    Döredildi!!! Forecast shablonda gowy cykar yaly!!!

    """

    cloudy = ['few clouds', 'scattered clouds', 'broken clouds', 'overcast clouds']
    foggy = ['mist', 'fog', 'smoke', 'haze']
    drizzle = [
        'light intensity drizzle',
        'drizzle',
        'heavy intensity drizzle',
        'light intensity drizzle rain',
        'drizzle rain',
        'heavy intensity drizzle rain',
        'shower rain and drizzle',
        'heavy shower rain and drizzle',
        'shower drizzle',
        'light rain',
        'moderate rain',
        'heavy intensity rain',
        'very heavy rain',
        'extreme rain',
        'freezing rain',
        'light intensity shower rain',
        'shower rain',
        'heavy intensity shower rain',
        'ragged shower rain',
    ]
    snow = [
        'light snow',
        'snow',
        'heavy snow',
        'sleet',
        'shower sleet',
        'light rain and snow',
        'rain and snow',
        'light shower snow',
        'shower snow',
        'heavy shower snow'
    ]
    sand = [
        'dust whirls',
        'sand',
        'dust',
        'volcanic ash',
    ]

    forecast_message = ""
    for date, data in formatted_forecast.items():
        min_temp = min(data['temperatures'])
        max_temp = max(data['temperatures'])
        description = data['descriptions'][0]  # берем описание только из первого времени дня

        if description in cloudy:
            forecast_message += f"Дата: {date}, Температура: {min_temp}°C - {max_temp}°C<br>"\
                                f"Облачная и пасмурная погода создает<br>" \
                                f"уютную атмосферу для посещения кафе или уютного места для чтения. <br>" \
                                f"Можно насладиться чашечкой горячего напитка, читать книгу или просто<br>" \
                                f"наслаждаться спокойным временем.<br>" \
                                f"можно выбрать легкую куртку или свитер. Они помогут защитить от прохлады, <br>" \
                                f"но при этом не будут слишком громоздкими. <br><br>"

        elif description in foggy:
            forecast_message += f"Дата: {date}, Температура: {min_temp}°C - {max_temp}°C<br>"\
                                f"Это может показаться не слишком захватывающим занятием, <br>" \
                                f"но уборка и организация дома могут приносить удовлетворение и создавать уютную обстановку. <br>" \
                                f"Попробуйте обновить свой интерьер или организовать свои вещи.<br>" \
                                f"Из-за ограниченной видимости важно, чтобы вас было легко заметить. <br>" \
                                f"Поэтому выбирайте одежду ярких цветов или с отражающими элементами, <br>" \
                                f"которые помогут вам быть более видимым в условиях тумана.<br>" \
                                f"Туман часто сопровождается повышенной влажностью и прохладой, поэтому выбирайте теплую одежду. <br>" \
                                f"Пальто, длинные пальто или куртки с утеплителем будут хорошим выбором. <br><br>"

        elif description in drizzle:
            forecast_message += f"Дата: {date}, Температура: {min_temp}°C - {max_temp}°C<br>"\
                                f"Устройте небольшой киномарафон с вашими любимыми фильмами или сериалами. <br>" \
                                f"Заварите себе чашку ароматного чая и наслаждайтесь просмотром. <br>" \
                                f"Одежда с водонепроницаемым покрытием будет хорошим выбором для защиты от дождя. <br>" \
                                f"Выберите легкий и компактный дождевик или плащ, <br>" \
                                f"который можно легко надеть поверх вашей основной одежды.<br><br>"

        elif description in snow:
            forecast_message += f"Дата: {date}, Температура: {min_temp}°C - {max_temp}°C<br>"\
                                f"Для саморазвития и обучения можно изучать новые темы онлайн, <br>" \
                                f"писать статьи или просто наслаждаться уединенным временем, обдумывая свои мысли и идеи.<br>" \
                                f" Наденьте утепленную куртку или пальто с ветрозащитным материалом, чтобы защититься от холода и ветра.<br>" \
                                f"Шапка, шарф и перчатки: Эти аксессуары помогут сохранить тепло и защитить от потери тепла через голову и руки. <br><br>"

        elif description in sand:
            forecast_message += f"Дата: {date}, Температура: {min_temp}°C - {max_temp}°C<br>"\
                                f"Уютное чтение — отличный способ провести время. Выберите книгу, <br>" \
                                f"которую давно хотели прочитать, или журнал по интересующей вас теме.<br>" \
                                f"При погоде с пылью, песком, пылевыми вихрями или вулканическим пеплом рекомендуется надеть защитную одежду и аксессуары. <br>" \
                                f"Носите защитные очки или солнцезащитные очки, <br>" \
                                f"широкополую шляпу или панаму для защиты глаз и лица от пыли и песка. <br>" \
                                f"Используйте маску, бандану или повязку, чтобы закрыть рот и нос и предотвратить вдыхание пыли и мелких частиц. <br>" \
                                f"Выбирайте светлую, свободносидящую одежду с длинными рукавами и брюками, <br>" \
                                f"чтобы полностью покрыть тело и минимизировать контакт с частицами. <br><br>" \

        elif description == 'clear sky':
            forecast_message += f"Дата: {date}, Температура: {min_temp}°C - {max_temp}°C<br>"\
                                f"Если у вас есть фотоаппарат или смартфон с хорошей камерой, <br>" \
                                f"отправляйтесь на прогулку и делайте красивые фотографии природы, <br>" \
                                f"архитектуры или просто своих близких.<br>" \
                                f"Носите легкую рубашку, майку или футболку из натуральных материалов, <br>" \
                                f"таких как хлопок или лен. Это поможет вам оставаться прохладным и комфортным в течение дня. <br><br>" \

        elif description == 'tornado':
            forecast_message += f"Дата: {date}, Температура: {min_temp}°C - {max_temp}°C<br>"\
                                f" Если вы находитесь в помещении, возьмите на себя плотную одежду, <br>" \
                                f"которая защитит от обломков и разрушений.<br>" \
                                f"Когда погода становится неблагоприятной из-за торнадо, ваша безопасность становится главным приоритетом. <br>" \
                                f"Поскольку в таких ситуациях важно оставаться в безопасном месте и следить за метеорологической обстановкой, <br>" \
                                f"заняться чем-то вряд ли получится. <br><br>" \

        # forecast_message += f"Дата: {date} Описание: {description}, Температура: {min_temp}°C - {max_temp}°C\n"
    return forecast_message


from fuzzywuzzy import process


def extract_location(query):

    """

    Döredildi! Query-dan diňe şäheriň adyny alar ýaly!!!

    """

    words = query.split()
    city_names = [
        "Mary", "mary", "Lebap", "lebap", "Ashgabat", "ashgabat", "Ahal", "ahal",
        "Balkan", "balkan", "Dashoguz", "dashoguz", "Мары", "мары", "Ашхабад",
        "ашхабад", "Лебап", "лебап", "Балкан", "балкан", "Дашогуз", "дашогуз", "Ахал", "ахал"
    ]

    matching_cities = []
    for word in words:
        matches = process.extract(word, city_names, limit=1)
        if matches and matches[0][1] >= 70:
            matching_cities.append(matches[0][0])

    matching_cities_str = ' '.join(matching_cities)

    return matching_cities_str


def save_dialogue(request, session_key, user_message, bot_response):

    """

    !!!
    Döredildi. Çatlary tekst faýla sohranit etýär
    !!!

    """

    # Define dialogue file path
    file_path = os.path.join('dialogues', f'{request.user}_{session_key}.txt')

    # Write dialogue to file
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(f"User: {user_message}\n")
        file.write(f"Bot: {bot_response}\n")
        file.write('\n')


def start_new_chat(request):

    """

    !!!
    Döredildi. Täze çata başlar ýaly sessiýa açaryňy çalyşýar
    !!!

    """

    request.session.cycle_key()  # Сбрасываем текущий ключ сеанса
    request.session.save()  # Сохраняем сеанс, чтобы обновить ключ

    # Перенаправляем пользователя на страницу с чатом
    return redirect('chat')


# В представлении для отображения деталей диалога
def dialogue_detail(request, user, session_key):

    """

    !!!
    Döredildi. Sohranit edilen çaty görer ýaly we täze sorag botdan sorar ýaly
    !!!

    """

    if request.user.is_authenticated:

        username = request.user.username
        email = request.user.email

        dialogue_file_path = os.path.join('dialogues', f'{user}_{session_key}.txt')

        print(dialogue_file_path)

        if os.path.exists(dialogue_file_path):
            with open(dialogue_file_path, 'r', encoding='utf-8') as file:
                messages = file.readlines()
                formatted_messages = []

                for message in messages:
                    if message.startswith('User:'):
                        formatted_messages.append(('user', message.split(':')[1].strip()))
                    elif message.startswith('Bot:'):
                        formatted_messages.append(('bot', message.split(':')[1].strip()))
        else:
            formatted_messages = []

        dialogue_files = os.listdir('dialogues')
        dialogues_with_session_keys = []
        current_user = request.user.username
        for dialogue_file in dialogue_files:
            file_username = dialogue_file.split('_')[0]
            if file_username == current_user:
                session_key = dialogue_file.split('_')[1].split('.')[0]
                with open(os.path.join('dialogues', dialogue_file), 'r', encoding='utf-8') as file:
                    first_line = file.readline().strip().replace('User: ', '')
                dialogues_with_session_keys.append((first_line, session_key))

        return render(request, 'chatbot/dialogue_detail.html', {'messages': formatted_messages, 'path': dialogue_file_path, 'email': email, 'username': username, 'dialogues_with_session_keys': dialogues_with_session_keys})
    else:
        return redirect('login')


def save_to_file(request):
    """

    !!!
    Üýtgedildi. Current weather and forecast goşuldy!!!
    !!!

    """

    if request.method == 'GET':
        query = request.GET.get('query', None)
        file_path = request.GET.get('file_path', None)
        file_path = file_path.replace('dialogues', 'dialogues/')
        print(file_path)
        if query:
            preprocessed_query = preprocess_text(query)

            tags = extract_tags(query)

            query_vectorized = vectorizer.transform([preprocessed_query])

            predicted_address = model.predict(query_vectorized)[0]

            data_values1 = ' '.join(data['question'].values)

            data_values2 = data_values1.split()

            x = any(i in preprocessed_query for i in data_values2)

            weather_keywords = ["weather", "howa", "погода"]
            forecast_keywords = ["прогноз погоды", "forecast", "howa maglumaty"]
            city_names = [
                "Mary", "mary", "Lebap", "lebap", "Ashgabat", "ashgabat", "Ahal", "ahal",
                "Balkan", "balkan", "Dashoguz", "dashoguz", "Мары", "мары", "Ашхабад",
                "ашхабад", "Лебап", "лебап", "Балкан", "балкан", "Дашогуз", "дашогуз", "Ахал", "ахал"
            ]

            has_weather_keyword = any(keyword in preprocessed_query for keyword in weather_keywords)
            has_forecast_keyword = any(forecast in preprocessed_query for forecast in forecast_keywords)
            has_city_name = any(city_name in preprocessed_query for city_name in city_names)

            if has_weather_keyword and has_city_name and not has_forecast_keyword:
                # Извлечение города из запроса
                location = extract_location(query)
                if location:
                    # Получаем местоположение из запроса (например, "погода в Мары")
                    # location = extract_location(preprocessed_query)
                    print(location)
                    # Запрос на прогноз погоды к API OpenWeatherMap
                    api_key = '7bb75bca252ee310e90e9a127ca642e2'
                    api_url = f'http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric'
                    print(api_url)
                    response = requests.get(api_url)
                    if response.status_code == 200:
                        weather_data = response.json()
                        # Обрабатываем данные о погоде и возвращаем ответ
                        weather_description = weather_data['weather'][0]['description']
                        temperature = weather_data['main']['temp']

                        ###############!!!!!!!!!!!!!##################

                        weather_response = ''

                        cloudy = ['few clouds', 'scattered clouds', 'broken clouds', 'overcast clouds']
                        foggy = ['mist', 'fog', 'smoke', 'haze']
                        drizzle = [
                            'light intensity drizzle',
                            'drizzle',
                            'heavy intensity drizzle',
                            'light intensity drizzle rain',
                            'drizzle rain',
                            'heavy intensity drizzle rain',
                            'shower rain and drizzle',
                            'heavy shower rain and drizzle',
                            'shower drizzle',
                            'light rain',
                            'moderate rain',
                            'heavy intensity rain',
                            'very heavy rain',
                            'extreme rain',
                            'freezing rain',
                            'light intensity shower rain',
                            'shower rain',
                            'heavy intensity shower rain',
                            'ragged shower rain',
                        ]
                        snow = [
                            'light snow',
                            'snow',
                            'heavy snow',
                            'sleet',
                            'shower sleet',
                            'light rain and snow',
                            'rain and snow',
                            'light shower snow',
                            'shower snow',
                            'heavy shower snow'
                        ]

                        sand = [
                            'dust whirls',
                            'sand',
                            'dust',
                            'volcanic ash',
                        ]

                        if weather_description in cloudy:
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                               f"Облачная и пасмурная погода создает <br>" \
                                               f"уютную атмосферу для посещения кафе или уютного места для чтения. <br>" \
                                               f"Можно насладиться чашечкой горячего напитка, читать книгу или просто <br>" \
                                               f"наслаждаться спокойным временем. <br>" \
                                               f"можно выбрать легкую куртку или свитер. Они помогут защитить от прохлады, <br>" \
                                               f"но при этом не будут слишком громоздкими. <br>"

                        elif weather_description in foggy:
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                                f"Это может показаться не слишком захватывающим занятием, <br>" \
                                                f"но уборка и организация дома могут приносить удовлетворение и создавать уютную обстановку. <br>" \
                                                f"Попробуйте обновить свой интерьер или организовать свои вещи.<br>"\
                                                f"Из-за ограниченной видимости важно, чтобы вас было легко заметить. <br>" \
                                                f"Поэтому выбирайте одежду ярких цветов или с отражающими элементами, <br>" \
                                                f"которые помогут вам быть более видимым в условиях тумана. <br>"\
                                                f"Туман часто сопровождается повышенной влажностью и прохладой, поэтому выбирайте теплую одежду. <br>" \
                                                f"Пальто, длинные пальто или куртки с утеплителем будут хорошим выбором."

                        elif weather_description in drizzle:
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                                f"Устройте небольшой киномарафон с вашими любимыми фильмами или сериалами. <br>" \
                                                f"Заварите себе чашку ароматного чая и наслаждайтесь просмотром. <br>" \
                                                f"Одежда с водонепроницаемым покрытием будет хорошим выбором для защиты от дождя. <br>" \
                                                f"Выберите легкий и компактный дождевик или плащ, <br>" \
                                                f"который можно легко надеть поверх вашей основной одежды."

                        elif weather_description in snow:
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>"\
                                                f"Для саморазвития и обучения можно изучать новые темы онлайн, <br>" \
                                                f"писать статьи или просто наслаждаться уединенным временем, обдумывая свои мысли и идеи.<br>"\
                                                f" Наденьте утепленную куртку или пальто с ветрозащитным материалом, чтобы защититься от холода и ветра.<br>" \
                                                f"Шапка, шарф и перчатки: Эти аксессуары помогут сохранить тепло и защитить от потери тепла через голову и руки."

                        elif weather_description in sand:
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                                f"Уютное чтение — отличный способ провести время. Выберите книгу, <br>" \
                                                f"которую давно хотели прочитать, или журнал по интересующей вас теме. <br>" \
                                                f"При погоде с пылью, песком, пылевыми вихрями или вулканическим пеплом рекомендуется надеть защитную одежду и аксессуары. <br>" \
                                                f"Носите защитные очки или солнцезащитные очки, <br>" \
                                                f"широкополую шляпу или панаму для защиты глаз и лица от пыли и песка. <br>" \
                                                f"Используйте маску, бандану или повязку, чтобы закрыть рот и нос и предотвратить вдыхание пыли и мелких частиц. <br>" \
                                                f"Выбирайте светлую, свободносидящую одежду с длинными рукавами и брюками, <br>" \
                                                f"чтобы полностью покрыть тело и минимизировать контакт с частицами."\

                        elif weather_description == 'clear sky':
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                                f"Если у вас есть фотоаппарат или смартфон с хорошей камерой, <br>" \
                                                f"отправляйтесь на прогулку и делайте красивые фотографии природы, <br>" \
                                                f"архитектуры или просто своих близких. <br>" \
                                                f"Носите легкую рубашку, майку или футболку из натуральных материалов, <br>" \
                                                f"таких как хлопок или лен. Это поможет вам оставаться прохладным и комфортным в течение дня." \

                        elif weather_description == 'tornado':
                            weather_response += f"Сейчас в {location} Температура {temperature}°C <br>" \
                                                f" Если вы находитесь в помещении, возьмите на себя плотную одежду, <br>" \
                                                f"которая защитит от обломков и разрушений. <br>" \
                                                f"Когда погода становится неблагоприятной из-за торнадо, ваша безопасность становится главным приоритетом. <br>" \
                                                f"Поскольку в таких ситуациях важно оставаться в безопасном месте и следить за метеорологической обстановкой, <br>" \
                                                f"заняться чем-то вряд ли получится." \

                        #####################!!!!!!!!!!!!!!!!!!!!!!!#############################

                        with open(file_path, 'a', encoding='utf-8') as file:
                            file.write(f"User: {query}\n")
                            file.write(f"Bot: {weather_response}\n")
                            file.write('\n')
                        return JsonResponse(
                            {'address': weather_response}
                        )
                    else:
                        error_response = 'Ошибка при получении прогноза погоды'
                        with open(file_path, 'a', encoding='utf-8') as file:
                            file.write(f"User: {query}\n")
                            file.write(f"Bot: {error_response}\n")
                            file.write('\n')
                        return JsonResponse({'address': error_response}, status=500)
                else:
                    default_response = "Извините, не могу определить местоположение для получения погоды."
                    with open(file_path, 'a', encoding='utf-8') as file:
                        file.write(f"User: {query}\n")
                        file.write(f"Bot: {default_response}\n")
                        file.write('\n')
                    return JsonResponse({'address': default_response})
            elif has_weather_keyword and not has_city_name:
                default_response = "В каком городе вы хотите узнать погоду?"
                with open(file_path, 'a', encoding='utf-8') as file:
                    file.write(f"User: {query}\n")
                    file.write(f"Bot: {default_response}\n")
                    file.write('\n')
                return JsonResponse({'address': default_response})

            elif has_forecast_keyword and has_city_name:
                location = extract_location(query)
                if location:
                    print(location)
                    # Запрос на прогноз погоды к API OpenWeatherMap
                    api_key = '7bb75bca252ee310e90e9a127ca642e2'
                    api_url = f'http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}&units=metric'
                    print(api_url)
                    response = requests.get(api_url)

                    if response.status_code == 200:
                        weather_data = response.json()

                        formatted_forecast = format_weather_forecast(weather_data)
                        forecast_message = format_forecast_message(formatted_forecast)

                        with open(file_path, 'a', encoding='utf-8') as file:
                            file.write(f"User: {query}\n")
                            file.write(f"Bot: {forecast_message}\n")
                            file.write('\n')
                        return JsonResponse({'address': forecast_message})

                    else:
                        error_response = 'Ошибка при получении прогноза погоды'
                        with open(file_path, 'a', encoding='utf-8') as file:
                            file.write(f"User: {query}\n")
                            file.write(f"Bot: {error_response}\n")
                            file.write('\n')
                        return JsonResponse({'address': error_response}, status=500)

                else:
                    default_response = "Извините, не могу определить местоположение для получения погоды."
                    with open(file_path, 'a', encoding='utf-8') as file:
                        file.write(f"User: {query}\n")
                        file.write(f"Bot: {default_response}\n")
                        file.write('\n')
                    return JsonResponse({'address': default_response})

            elif has_weather_keyword and not has_city_name:
                default_response = "В каком городе вы хотите узнать прогноз погоды?"
                with open(file_path, 'a', encoding='utf-8') as file:
                    file.write(f"User: {query}\n")
                    file.write(f"Bot: {default_response}\n")
                    file.write('\n')
                return JsonResponse({'address': default_response})

            ###################################################################

            elif tags and not x:
                if '<дата>' in tags:
                    current_date = datetime.date.today()
                    date = f"Сегодняшняя дата {current_date}"
                    with open(file_path, 'a', encoding='utf-8') as file:
                        file.write(f"User: {query}\n")
                        file.write(f"Bot: {date}\n")
                        file.write('\n')
                    return JsonResponse({'address': date})
                elif '<время>' in tags:
                    tz_ashgabat = pytz.timezone('Asia/Ashgabat')
                    current_time = datetime.datetime.now(tz=tz_ashgabat).strftime('%H-%M')
                    time = f"Текущее время {current_time}"
                    with open(file_path, 'a', encoding='utf-8') as file:
                        file.write(f"User: {query}\n")
                        file.write(f"Bot: {time}\n")
                        file.write('\n')
                    return JsonResponse({'address': time})
                elif '<день_недели>' in tags:
                    lang = detect_language(query)
                    print(lang)
                    with open(file_path, 'a', encoding='utf-8') as file:
                        file.write(f"User: {query}\n")
                        file.write(f"Bot: {get_weekday(lang)}\n")
                        file.write('\n')
                    return JsonResponse({'address': get_weekday(lang)})

                ##############################################################

            else:
                if x:
                    with open(file_path, 'a', encoding='utf-8') as file:
                        file.write(f"User: {query}\n")
                        file.write(f"Bot: {predicted_address}\n")
                        file.write('\n')
                    return JsonResponse({'address': predicted_address})
                else:
                    default_response = "Извините, не могу найти ответ на ваш вопрос."
                    with open(file_path, 'a', encoding='utf-8') as file:
                        file.write(f"User: {query}\n")
                        file.write(f"Bot: {default_response}\n")
                        file.write('\n')
                    return JsonResponse({'address': default_response})

        else:
            return JsonResponse({'error': 'Query parameter is missing'}, status=400)


def delete_dialogue(request, session_key):
    """

    Döredildi! Dialogy udalit eder ýaly

    """
    dialogue_file_path = os.path.join('dialogues', f'{request.user}_{session_key}.txt')
    print(dialogue_file_path)
    if request.method == 'POST' and 'confirm_delete' in request.POST:
        # if os.path.exists(dialogue_file_path):
        os.remove(dialogue_file_path)  # Удаляем файл
        return redirect('chat')
    return render(request, 'chatbot/user_dialogues_delete.html', {'session_key': session_key})


# Autorization logic

def signup(request):
    if request.method == 'POST':
        form = SignupForm(request.POST, request.FILES)
        if form.is_valid():
            # save form in the memory not in database
            user = form.save(commit=False)
            user.is_active = False
            user.save()
            # to get the domain of the current site
            current_site = get_current_site(request)
            mail_subject = 'Activation link has been sent to your email id'
            message = render_to_string('chatbot/acc_active_email.html', {
                'user': user,
                'domain': current_site.domain,
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'token': account_activation_token.make_token(user),
            })
            to_email = form.cleaned_data.get('email')
            email = EmailMessage(
                mail_subject, message, to=[to_email]
            )
            email.send()
            # return HttpResponse('Please confirm your email address to complete the registration')
            return redirect('confirm')
    else:
        form = SignupForm()
    return render(request, 'chatbot/signup.html', {'form': form})


def confirm(request):
    return render(request, 'chatbot/confirm.html')


def success(request):
    return render(request, 'chatbot/success.html')


def activate(request, uidb64, token):
    User = get_user_model()
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except(TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.save()
        login(request, user)
        # return HttpResponse('Thank you for your email confirmation. Now you can login your account.')
        return redirect('success')
    else:
        return HttpResponse('Activation link is invalid!')


def signout(request):
    logout(request)
    return redirect('register')


def user_login(request):
    if request.method == 'POST':
        form = UserLoginForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = UserLoginForm()
    return render(request, 'chatbot/login.html', {'form': form})


class CustomPasswordResetView(PasswordResetView):
    template_name = 'chatbot/custom_reset_password.html'


class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    template_name = 'chatbot/custom_password_reset_confirm.html'


class CustomPasswordResetDoneView(PasswordResetDoneView):
    template_name = 'chatbot/confirm.html'


class CustomPasswordResetCompleteView(PasswordResetCompleteView):
    template_name = 'chatbot/custom_password_reset_complete.html'


def news(request):
    news = News.objects.all()
    paginator = Paginator(news, 10)
    page_num = request.GET.get('page')
    page_obj = paginator.get_page(page_num)
    return render(request, 'chatbot/news.html', {'page_obj': page_obj, "news": news})


def news_detail(request, pk):
    news_detail = News.objects.get(pk=pk)
    return render(request, 'chatbot/news_detail.html', {'news_detail': news_detail})


def about(request):
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            feedback = form.save()
            subject = form.cleaned_data['subject']
            message = f'Email: {feedback.email}\nMessage: {feedback.subject}'
            from_email = form.cleaned_data['email']
            to_email = settings.DEFAULT_FROM_EMAIL
            mail = send_mail(subject, message, from_email, [to_email], fail_silently=False)
            if mail:
                form.save()
                messages.success(request, _('Hat üstünlikli ugradyldy'))
            else:
                feedback.delete()
                messages.error(request, _('Internet birikdirmesi ýitdi'))
        else:
            messages.error(request, _('Näsazlyk ýüze çykdy'))
            # return redirect('about')
    else:
        form = FeedbackForm()
    return render(request, 'chatbot/about.html', {'form': form})


"""

Рассылка!!!


"""


@user_passes_test(lambda u: u.is_staff)
def send_newsletter(request):
    if request.method == 'POST':
        form = NewsletterForm(request.POST, request.FILES)
        if form.is_valid():
            subject = form.cleaned_data['subject']
            message = form.cleaned_data['message']
            # recipients = User.objects.filter(is_active=True).values_list('email', flat=True)
            recipients = Subscribes.objects.all()
            # HTML-разметка для красивого письма
            html_message = f"""
            <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Awesome Newsletter</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: deepskyblue;
            margin: 0;
            padding: 0;
        }}

        .container {{
            max-width: 600px;
            margin: 20px auto;
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}

        h2 {{
            color: #3498db;
        }}

        p {{
            font-size: 16px;
            color: #333;
        }}

        .button-container {{
            text-align: center;
            margin-top: 20px;
        }}

        .button {{
            display: inline-block;
            padding: 10px 20px;
            font-size: 18px;
            color: #fff;
            text-decoration: none;
            border-radius: 5px;
            background-color: #3498db;
            transition: background-color 0.3s ease;
        }}

        .button:hover {{
            background-color: #2980b9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>{subject}</h2>
        <p>{message}</p>

        <img src="https://lh3.googleusercontent.com/proxy/vOL5VUrTwGQDjw_HI-fogWgpk0dxaDUAWozIO9zNFun9kfTUHeJrifv3XmlMUse6RjP7_YlXqy9yF3KzsT1f2s8nDJQsRfK79zioKIYe">
        <div class="button-container">
            <a href="http://shatumar.com.tm/ru/" class="button">Сделай заказ сейчас</a>
        </div>
    </div>
</body>
</html>

            """

            try:
                send_mail(
                    subject,
                    message,
                    settings.DEFAULT_FROM_EMAIL,
                    recipients,
                    html_message=html_message,
                )
                return HttpResponse('Success')
            except Exception as e:
                # Обработка ошибок при отправке электронной почты
                print(e)
                return HttpResponse('Error')

    else:
        form = NewsletterForm()
    return render(request, 'chatbot/send_newsletter.html', {'form': form})
