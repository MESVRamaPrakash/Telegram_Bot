import telebot
import requests
import google.generativeai as genai
import datetime
import time
import subprocess
import img2pdf
import os
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch
from gtts import gTTS
import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from googletrans import Translator
import nltk
nltk.download('punkt')
# from diffusers import DiffusionPipeline
#
# pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-cascade")


# API_URL = "https://github.com/Stability-AI/generative-models"
# API_TOKEN_HF = "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"

# Set up your API keys
TelegramBOT_TOKEN = ''
ANTHROPIC_API_KEY = ''
GENERATIVE_AI_KEY = ''
TMDB_API_KEY = ''
OPENWEATHER_API_KEY = ''
NEWS_API_KEY = ''
NASA_API_KEY = ''

CAT_API_URL = "https://api.thecatapi.com/v1/images/search"
POKEAPI_URL = "https://pokeapi.co/api/v2/pokemon/"
# Configure generative AI
genai.configure(api_key=GENERATIVE_AI_KEY)





# Initialize the conversation model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Initialize the conversation
convo = model.start_chat(history=[])
# Ensure necessary directories exist
if not os.path.exists('downloads'):
    os.makedirs('downloads')
if not os.path.exists('images'):
    os.makedirs('images')
if not os.path.exists('pdfs'):
    os.makedirs('pdfs')

bot = telebot.TeleBot(TelegramBOT_TOKEN)

# Modify the recognizer initialization
recognizer = sr.Recognizer()


tts_engine = pyttsx3.init()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Dictionary containing symptoms and associated diseases
symptom_disease_mapping = {
    'abdominal pain': ['appendicitis', 'gastroenteritis', 'ulcer', 'kidney stones'],
    'back pain': ['muscle strain', 'herniated disc', 'kidney stones', 'spinal stenosis'],
    'cough': ['common cold', 'flu', 'pneumonia', 'bronchitis'],
    'dizziness': ['vertigo', 'migraine', 'inner ear infection', 'dehydration'],
    'earache': ['ear infection', 'earwax buildup', 'TMJ disorder', 'sinus infection'],
    'fever': ['flu', 'pneumonia', 'urinary tract infection', 'sinus infection'],
    'headache': ['tension headache', 'migraine', 'cluster headache', 'sinus headache'],
    'itchy skin': ['eczema', 'psoriasis', 'allergic reaction', 'scabies'],
    'joint pain': ['arthritis', 'gout', 'bursitis', 'Lyme disease'],
    'knee pain': ['osteoarthritis', 'meniscus tear', 'ligament injury', 'patellar tendinitis'],
    'lightheadedness': ['orthostatic hypotension', 'anxiety', 'dehydration', 'iron deficiency anemia'],
    'muscle weakness': ['muscular dystrophy', 'myasthenia gravis', 'hypothyroidism', 'fibromyalgia'],
    'nausea': ['food poisoning', 'viral gastroenteritis', 'migraine', 'pregnancy'],
    'rash': ['contact dermatitis', 'eczema', 'psoriasis', 'allergic reaction'],
    'shortness of breath': ['asthma', 'chronic obstructive pulmonary disease (COPD)', 'pneumonia', 'heart failure'],
    'toothache': ['tooth decay', 'gum disease', 'tooth abscess', 'sinus infection'],
    'vomiting': ['food poisoning', 'viral gastroenteritis', 'migraine', 'pregnancy'],
    'weakness': ['anemia', 'chronic fatigue syndrome', 'hypothyroidism', 'fibromyalgia'],
    'abnormal menstruation': ['polycystic ovary syndrome', 'endometriosis', 'uterine fibroids', 'thyroid disorder'],
    'acne': ['hormonal changes', 'polycystic ovary syndrome', 'cushing syndrome', 'medications'],
    'anxiety': ['generalized anxiety disorder', 'panic disorder', 'social anxiety disorder', 'PTSD'],
    'blurred vision': ['refractive errors', 'cataracts', 'glaucoma', 'diabetic retinopathy'],
    'bloating': ['irritable bowel syndrome', 'celiac disease', 'lactose intolerance', 'constipation'],
    'bruising': ['leukemia', 'hemophilia', 'liver disease', 'vitamin K deficiency'],
    'chest pain': ['heart attack', 'angina', 'gastroesophageal reflux disease (GERD)', 'pulmonary embolism'],
    'chills': ['flu', 'malaria', 'urinary tract infection', 'mononucleosis'],
    'constipation': ['irritable bowel syndrome', 'hypothyroidism', 'dehydration', 'medications'],
    'diarrhea': ['gastroenteritis', 'food poisoning', 'irritable bowel syndrome', 'celiac disease'],
    'difficulty swallowing': ['GERD', 'esophagitis', 'stroke', 'esophageal cancer'],
    'dry mouth': ['dehydration', 'Sjogrens syndrome', 'diabetes', 'medications'],
    'dry skin': ['eczema', 'psoriasis', 'hypothyroidism', 'dehydration'],
    'fatigue': ['anemia', 'chronic fatigue syndrome', 'depression', 'sleep apnea'],
    'flushing': ['menopause', 'rosacea', 'carcinoid syndrome', 'medications'],
    'frequent urination': ['diabetes', 'urinary tract infection', 'prostate issues', 'diuretics'],
    'hair loss': ['alopecia areata', 'thyroid disorders', 'nutritional deficiencies', 'medications'],
    'heart palpitations': ['arrhythmia', 'anxiety', 'hyperthyroidism', 'anemia'],
    'hiccups': ['gastroesophageal reflux disease (GERD)', 'stroke', 'pleurisy', 'kidney failure'],
    'high blood pressure': ['hypertension', 'kidney disease', 'Cushings syndrome', 'sleep apnea'],
    'hoarseness': ['laryngitis', 'vocal cord nodules', 'thyroid problems', 'GERD'],
    'indigestion': ['gastroesophageal reflux disease (GERD)', 'ulcer', 'gastroparesis', 'lactose intolerance'],
    'irregular heartbeat': ['arrhythmia', 'hyperthyroidism', 'heart attack', 'electrolyte imbalance'],
    'jaundice': ['hepatitis', 'liver cirrhosis', 'gallstones', 'pancreatic cancer'],
    'joint swelling': ['arthritis', 'gout', 'bursitis', 'infection'],
    'loss of appetite': ['depression', 'chronic liver disease', 'kidney failure', 'cancer'],
    'low blood pressure': ['hypotension', 'Addisons disease', 'dehydration', 'heart problems'],
    'muscle cramps': ['dehydration', 'electrolyte imbalance', 'muscle strain', 'neuropathy'],
    'night sweats': ['tuberculosis', 'lymphoma', 'hyperthyroidism', 'menopause'],
    'numbness': ['neuropathy', 'stroke', 'multiple sclerosis', 'diabetes'],
    'painful urination': ['urinary tract infection', 'sexually transmitted infection', 'bladder stones', 'prostatitis'],
    'persistent cough': ['chronic bronchitis', 'lung cancer', 'asthma', 'GERD'],
    'rapid heartbeat': ['tachycardia', 'anemia', 'anxiety', 'hyperthyroidism'],
    'ringing in ears': ['tinnitus', 'hearing loss', 'Meniere’s disease', 'ear injury'],
    'runny nose': ['common cold', 'allergies', 'sinusitis', 'nasal polyps'],
    'seizures': ['epilepsy', 'brain injury', 'infection', 'stroke'],
    'shortness of breath': ['asthma', 'COPD', 'heart failure', 'pulmonary embolism'],
    'skin rash': ['allergic reaction', 'eczema', 'psoriasis', 'contact dermatitis'],
    'sore throat': ['pharyngitis', 'tonsillitis', 'laryngitis', 'mononucleosis'],
    'stomach cramps': ['gastroenteritis', 'irritable bowel syndrome', 'food poisoning', 'menstrual cramps'],
    'sweating': ['hyperthyroidism', 'menopause', 'infection', 'anxiety'],
    'swelling': ['lymphedema', 'infection', 'heart failure', 'kidney disease'],
    'swollen glands': ['infection', 'mononucleosis', 'lymphoma', 'HIV/AIDS'],
    'tingling': ['neuropathy', 'multiple sclerosis', 'stroke', 'carpal tunnel syndrome'],
    'tremor': ['Parkinson’s disease', 'essential tremor', 'anxiety', 'hyperthyroidism'],
    'trouble sleeping': ['insomnia', 'sleep apnea', 'depression', 'anxiety'],
    'unexplained weight loss': ['cancer', 'hyperthyroidism', 'diabetes', 'chronic infection'],
    'urinary incontinence': ['stress incontinence', 'urge incontinence', 'overflow incontinence', 'functional incontinence'],
    'vision problems': ['cataracts', 'glaucoma', 'macular degeneration', 'diabetic retinopathy'],
    'weakness': ['anemia', 'chronic fatigue syndrome', 'hypothyroidism', 'fibromyalgia'],
    'wheezing': ['asthma', 'COPD', 'allergic reaction', 'bronchitis'],
    'yellow skin': ['jaundice', 'hepatitis', 'liver cirrhosis', 'gallstones'],
    'abnormal vaginal bleeding': ['uterine fibroids', 'endometriosis', 'cervical cancer', 'hormonal imbalance'],
    'acid reflux': ['GERD', 'hiatal hernia', 'peptic ulcer', 'pregnancy'],
    'bleeding gums': ['gingivitis', 'periodontitis', 'vitamin deficiency', 'blood disorders'],
    'blood in urine': ['urinary tract infection', 'kidney stones', 'bladder cancer', 'prostate problems'],
    'blurred vision': ['diabetes', 'glaucoma', 'cataracts', 'refractive errors'],
    'burning sensation': ['neuropathy', 'urinary tract infection', 'shingles', 'chemical exposure'],
    'chest discomfort': ['heart attack', 'angina', 'GERD', 'anxiety'],
}
# Define Emoji Mapping
emoji_mapping = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'love': '❤️',
    'laugh': '😂',
    'cry': '😭',
    'cool': '😎',
    'surprised': '😮',
    'sleepy': '😴',
    'wink': '😉',
    'shock': '😱',
    'confused': '😕',
    'heart': '❤️',
    'thumbs up': '👍',
    'thumbs down': '👎',
    'fire': '🔥',
    'star': '⭐',
    'money': '💰',
    'rocket': '🚀',
    'clap': '👏',
    'pray': '🙏',
    'ghost': '👻',
    'unicorn': '🦄',
    'cake': '🍰',
    'pizza': '🍕',
    'beer': '🍺',
    'taco': '🌮',
    'burrito': '🌯',
    'coffee': '☕',
    'book': '📚',
    'pencil': '✏️',
    'computer': '💻',
    'telephone': '☎️',
    'house': '🏠',
    'car': '🚗',
    'bike': '🚲',
    'train': '🚆',
    'airplane': '✈️',
    'boat': '⛵',
    'umbrella': '☔',
    'sun': '☀️',
    'moon': '🌙',
    'star': '⭐',
    'cloud': '☁️',
    'rain': '🌧️',
    'snowflake': '❄️',
    'tornado': '🌪️',
    'earth': '🌍',
    'globe': '🌐',
    'map': '🗺️',
    'alarm clock': '⏰',
    'hourglass': '⌛',
    'watch': '⌚',
    'lock': '🔒',
    'key': '🔑',
    'mail': '📧',
    'package': '📦',
    'gift': '🎁',
    'birthday': '🎂',
    'party': '🎉',
    'christmas tree': '🎄',
    'jack-o-lantern': '🎃',
    'fireworks': '🎆',
    'balloon': '🎈',
    'game': '🎮',
    'musical note': '🎵',
    'microphone': '🎤',
    'headphones': '🎧',
    'camera': '📷',
    'video camera': '📹',
    'television': '📺',
    'radio': '📻',
    'flag': '🚩',
    'traffic light': '🚥',
    'construction': '🚧',
    'warning': '⚠️',
    'stop sign': '🛑',
    'female': '♀️',
    'male': '♂️',
    'baby': '👶',
    'child': '🧒',
    'adult': '👨',
    'elderly': '👵',
    'family': '👪',
    'couple': '👫',
    'person': '👤',
    'people': '👥',
    'thumbs up': '👍',
    'thumbs down': '👎',
    'heart': '❤️',
    'heartbreak': '💔',
    'kiss': '💋',
    'ring': '💍',
    'diamond': '💎',
    'shirt': '👕',
    'jeans': '👖',
    'dress': '👗',
    'bikini': '👙',
    'purse': '👛',
    'handbag': '👜',
    'sandal': '👡',
    'shoe': '👟',
    'boot': '👢',
    'crown': '👑',
    'lipstick': '💄',
    'eyeglasses': '👓',
    'tie': '👔',
    'scarf': '🧣',
    'gloves': '🧤',
    'coat': '🧥',
    'socks': '🧦',
    'bald': '👴',
    'bearded': '🧔',
    'wizard': '🧙‍♂️',
    'elf': '🧝‍♂️',
    'vampire': '🧛‍♂️',
    'zombie': '🧟‍♂️',
    'mermaid': '🧜‍♀️',
    'fairy': '🧚‍♀️',
    'genie': '🧞‍♂️',
    'superhero': '🦸'
}
#model = torch.hub.load('ultralytics/yolo', 'yolo')

# Handler for welcome messages
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Welcome! AI BOT from RAD.Use /image to text to image generation,Use /audio [text], Use /translate [text] for translation Use /summary [para] for text to summary conversion, Use /emoji [expression] example /emoji happy, Use /med [symptom] for diseases and symptoms Use /weather [city] to get weather info, /movie [name] to get movie info,/news [category] to get news,Use /neo to get info about Near Earth Objects. /book [title] to get book info, upload image to convert it to pdf, Use /cat for cat image . /pokemon_name [name] for pokemon. and more.")

# Function to generate video from text prompt
# def generate_video_from_text(prompt):
#     step = 4  # Options: [1,2,4,8]
#     repo = "ByteDance/AnimateDiff-Lightning"
#     ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
#     base = "emilianJR/epiCRealism"  # Choose your favorite base model.
#
#     adapter = MotionAdapter().to(device, dtype)
#     adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
#     pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
#     pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
#
#     output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=step)
#     export_to_gif(output.frames, "animation.gif")

# Function to perform object detection using YOLOv5
# def detect_objects(image_path):
#     results = model(image_path)
#     labels = results.names  # Get class labels
#     detections = results.xyxy[0].cpu().numpy()  # Get detection results
#     detected_labels = [labels[int(d[5])] for d in detections]  # Get detected object labels
#     return detected_labels
#text-to-image

# def generate_image_from_text(text):
#     try:
#         # Generate the image from text using the Diffusion pipeline
#         image = pipeline(text)
#         return image
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

def text_to_audio(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_file = 'output.mp3'
    tts.save(audio_file)
    return audio_file
def translate_text(text, dest_lang='en'):
    """
    Translate the given text to the specified destination language.

    Args:
        text (str): The text to be translated.
        dest_lang (str): The destination language code. Defaults to 'en' (English).

    Returns:
        str: The translated text.
    """
    translator = Translator()
    translated = translator.translate(text, dest=dest_lang)
    return translated.text

def summarize_text_sumy(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])
def recognize_speech_from_file(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        audio.export(file_path, format="wav")
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text.lower()
    except sr.UnknownValueError:
        return "Sorry, I did not understand that."
    except sr.RequestError as e:
        return f"Could not request results; {e}"

def speak(text):
    tts_engine.save_to_file(text, 'response.mp3')
    tts_engine.runAndWait()
    return 'response.mp3'


def generate_response(prompt):
    response = genai.generate_text(prompt=prompt)
    return response.result  # Adjust if necessary based on the API's response format


# Emoji Function
def get_emoji(text):
    return emoji_mapping.get(text.lower(), '❓')  # Default to question mark if no matching emoji

def get_diseases(symptoms):
    diseases = []
    for symptom in symptoms:
        symptom = symptom.lower()
        if symptom in symptom_disease_mapping:
            diseases.extend(symptom_disease_mapping[symptom])
    return set(diseases)  # Use set to avoid duplicate diseases
# Function to get news
def get_news(category):
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey=089f2552540c4e76a1e04445ec40c5ef"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        articles = data['articles']
        news = ''
        for article in articles:
            title = article['title']
            news += f"📰 {title}\n"
        return news
    else:
        return "Sorry, I couldn't retrieve the news. Please try again later."
# Function to get movie info from TMDb
def get_movie_info(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            movie = data['results'][0]
            title = movie['title']
            overview = movie['overview']
            release_date = movie['release_date']
            return f"Title: {title}\nRelease Date: {release_date}\nOverview: {overview}"
        else:
            return "No results found."
    else:
        return "Error fetching data."

# Function to get weather info
def get_weather_info(location):
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(weather_url)
    data = response.json()

    if response.status_code == 200:
        weather_desc = data['weather'][0]['description']
        temp = data['main']['temp']
        city = data['name']
        country = data['sys']['country']
        return f"Weather in {city}, {country}:\n{weather_desc.capitalize()}\nTemperature: {temp}°C"
    else:
        return "Sorry, I couldn't retrieve the weather information. Please check the location."

#nasa function
# Function to get Near Earth Objects data from NASA
def get_neo_data(start_date, end_date):
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={NASA_API_KEY}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        neo_info = []
        for date in data['near_earth_objects']:
            for neo in data['near_earth_objects'][date]:
                name = neo.get('name')
                close_approach_date = neo['close_approach_data'][0].get('close_approach_date')
                miss_distance = neo['close_approach_data'][0]['miss_distance'].get('kilometers')
                velocity = neo['close_approach_data'][0]['relative_velocity'].get('kilometers_per_hour')
                neo_info.append(f"Name: {name}\nDate: {close_approach_date}\nMiss Distance: {miss_distance} km\nVelocity: {velocity} km/h\n")
        return neo_info
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return ["Error fetching data from NASA NEO API."]

# Helper function to split messages
def split_message(message, max_length=4096):
    return [message[i:i+max_length] for i in range(0, len(message), max_length)]

# Function to get book info from Open Library
def get_book_info(book_title):
    url = f"https://openlibrary.org/search.json?title={book_title}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['docs']:
            book = data['docs'][0]
            title = book.get('title')
            author_name = ", ".join(book.get('author_name', []))
            first_publish_year = book.get('first_publish_year')
            cover_id = book.get('cover_i')
            cover_url = f"http://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else None
            return title, author_name, first_publish_year, cover_url
        else:
            return None, None, None, None
    else:
        return None, None, None, None

# Convert PDF to Image
def pdf_to_images(pdf_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use pdftoppm command-line tool to convert PDF to images
    subprocess.run(['pdftoppm', '-png', pdf_path, os.path.join(output_dir, 'page')])

    print("Conversion complete.")

# Convert Image to PDF
def image_to_pdf(image_paths, output_pdf_path):
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(image_paths))



# Cat API Functions
def get_random_cat_image():
    response = requests.get(CAT_API_URL)
    if response.status_code == 200:
        data = response.json()
        image_url = data[0]['url']
        return image_url
    else:
        return None

# PokeAPI Functions
def get_pokemon_info(pokemon_name):
    url = f"{POKEAPI_URL}{pokemon_name.lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Extract relevant information from the response
        # Return the information as a formatted string
    else:
        return "Pokemon not found."


# @bot.message_handler(commands=['video'])
# def video_handler(message):
#     try:
#         # Extract the text prompt from the message
#         prompt = message.text.split('/video', 1)[1].strip()
#
#         # Notify the user that the generation is in progress
#         bot.send_message(message.chat.id, "Generating video, please wait...")
#
#         # Generate the video from the text prompt
#         generate_video_from_text(prompt)
#
#         # Send the generated video back to the user
#         with open("animation.gif", "rb") as video_file:
#             bot.send_animation(message.chat.id, video_file)
#     except Exception as e:
#         bot.reply_to(message, f"An error occurred: {str(e)}")


@bot.message_handler(commands=["image"])
def handle_message(message):
    # Retrieve the message text
    text = message.text

    model_id = "dreamlike-art/dreamlike-photoreal-2.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    prompt = text
    image = pipe(prompt).images[0]
    bot.send_photo(message.chat.id, image)

# Command handler for /image
# @bot.message_handler(commands=['image'])
# def send_image(message):
#     text = message.text.replace('/image', '').strip()
#     if text:
#         # Generate the image
#         image = generate_image_from_text(text)
#         if image:
#             # Save the image to a temporary file
#             temp_image_path = 'temp_image.png'
#             image.save(temp_image_path)
#             # Send the image to the user
#             with open(temp_image_path, 'rb') as image_file:
#                 bot.send_photo(message.chat.id, image_file)
#             # Remove the temporary image file
#             os.remove(temp_image_path)
#         else:
#             bot.reply_to(message, "Failed to generate image. Please try again later.")
#     else:
#         bot.reply_to(message, "Please provide a description to generate an image. Usage: /image <your description>")

# Command handler for /upload
# @bot.message_handler(commands=['upload'])
# def upload_image(message):
#     bot.reply_to(message, "Please upload an image.")
#
# # Message handler for receiving images
# @bot.message_handler(content_types=['photo'])
# def handle_image(message):
#     try:
#         # Get the file ID of the uploaded image
#         file_id = message.photo[-1].file_id
#         # Download the image
#         file_info = bot.get_file(file_id)
#         downloaded_file = bot.download_file(file_info.file_path)
#         # Save the image to a temporary file
#         temp_image_path = 'temp_image.jpg'
#         with open(temp_image_path, 'wb') as image_file:
#             image_file.write(downloaded_file)
#         # Perform object detection
#         objects = detect_objects(temp_image_path)
#         if objects:
#             response = f"The image contains: {', '.join(objects)}"
#             bot.reply_to(message, response)
#         else:
#             bot.reply_to(message, "No objects detected in the image.")
#         # Remove the temporary image file
#         os.remove(temp_image_path)
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         bot.reply_to(message, "Failed to process the image. Please try again later.")

#text-audio handler
@bot.message_handler(commands=['audio'])
def handle_audio_command(message):
    text = message.text[len('/audio '):].strip()
    if not text:
        bot.reply_to(message, 'Please provide the text you want to convert  to audio.')
        return
    audio_file = text_to_audio(text)
    with open(audio_file, 'rb') as f:
        bot.send_audio(message.chat.id, f)
    os.remove(audio_file)

@bot.message_handler(commands=['translate'])
def handle_translate(message):
    try:
        text_to_translate = message.text[len('/translate '):].strip()
        if not text_to_translate:
            bot.reply_to(message, "Please provide the text you want to translate after the /translate command.")
            return

        translated_text = translate_text(text_to_translate, dest_lang='te')  # Translate to Telugu
        bot.reply_to(message, translated_text)

    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your request.")

@bot.message_handler(commands=['summary'])
def handle_summary(message):
    try:
        # Extract the text to summarize from the message
        text_to_summarize = message.text[len('/summary '):].strip()

        if not text_to_summarize:
            bot.reply_to(message, "Please provide the text you want to summarize after the /summary command.")
            return

        # Limit input size to 1000 characters to reduce data usage
        if len(text_to_summarize) > 1000:
            text_to_summarize = text_to_summarize[:1000] + "..."

        # Generate the summary
        summary = summarize_text_sumy(text_to_summarize)

        # Send the summary back to the user
        bot.reply_to(message, summary)

    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your request.")

@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    try:
        file_info = bot.get_file(message.voice.file_id)
        file = requests.get(f'https://api.telegram.org/file/bot{TelegramBOT_TOKEN}/{file_info.file_path}')

        with open('voice.ogg', 'wb') as f:
            f.write(file.content)

        user_input = recognize_speech_from_file('voice.ogg')
        response_text = generate_response(user_input)
        response_audio_path = speak(response_text)

        with open(response_audio_path, 'rb') as audio:
            bot.send_voice(message.chat.id, audio)

        os.remove('voice.ogg')
        os.remove(response_audio_path)
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {e}")
# Handler for emoji command
@bot.message_handler(commands=['emoji'])
def send_emoji(message):
    try:
        word = message.text.split(maxsplit=1)[1]  # Get word from command
        emoji = emoji_mapping.get(word.lower())
        if emoji:
            bot.reply_to(message, emoji)
        else:
            bot.reply_to(message, "Sorry, no emoji found for that word.")
    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your request. Please use the format /emoji [word]")

# Handler for medical command
@bot.message_handler(commands=['med'])
def send_disease_info(message):
    try:
        symptoms = message.text.split(maxsplit=1)[1].split(',')  # Get symptoms from command
        symptoms = [symptom.strip() for symptom in symptoms]  # Strip whitespace
        diseases = get_diseases(symptoms)
        if diseases:
            response = "Based on the symptoms, the possible diseases are:\n" + '\n'.join(diseases)
        else:
            response = "No associated diseases found for the given symptoms."
        bot.reply_to(message, response)
    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your request. Please use the format /med [symptoms]")


# Handler for news command
@bot.message_handler(commands=['news'])
def send_news(message):
    try:
        if message is None or message.text is None:
            raise ValueError("Invalid message")

        category = message.text.split(maxsplit=1)[1]  # Get news category from command
        news_report = get_news(category)
        bot.reply_to(message, news_report)
    except ValueError as ve:
        print(f"ValueError: {ve}")
        bot.reply_to(message, "Sorry, I couldn't process your request. Please try again.")
    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I encountered an error while processing your request.")

# Handler for weather command
@bot.message_handler(commands=['weather'])
def send_weather(message):
    try:
        location = message.text.split(maxsplit=1)[1]  # Get location from command
        weather_report = get_weather_info(location)
        bot.reply_to(message, weather_report)
    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your request. Please use the format /weather [city name]")

# Handler for movie command
@bot.message_handler(commands=['movie'])
def send_movie_info(message):
    try:
        movie_name = message.text.split(maxsplit=1)[1]  # Get movie name from command
        movie_report = get_movie_info(movie_name)
        bot.reply_to(message, movie_report)
    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your request. Please use the format /movie [movie name]")


# Handler for NASA NEO command
@bot.message_handler(commands=['neo'])
def send_neo_data(message):
    try:
        # Example date range, can be replaced with dynamic dates
        start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        neo_reports = get_neo_data(start_date, end_date)

        for report in neo_reports:
            messages = split_message(report)
            for msg in messages:
                bot.reply_to(message, msg)
                time.sleep(1)  # Avoid hitting Telegram rate limits

    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your request.")


@bot.message_handler(commands=['book'])
def send_book_info(message):
    try:
        book_title = message.text.split(maxsplit=1)[1]  # Get book title from command
        title, author_name, first_publish_year, cover_url = get_book_info(book_title)
        if title and author_name and first_publish_year:
            response = f"Title: {title}\nAuthor: {author_name}\nFirst Published: {first_publish_year}"
            bot.send_message(message.chat.id, response)
            if cover_url:
                bot.send_photo(message.chat.id, cover_url)
        else:
            bot.reply_to(message, "No results found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your request. Please use the format /book [book title]")


# Handler for documents (PDFs)
@bot.message_handler(content_types=['document'])
def handle_docs(message):
    try:
        # Download the PDF file
        file_info = bot.get_file(message.document.file_id)
        file_path = os.path.join('downloads', file_info.file_path)
        file_url = f'https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}'
        subprocess.run(['wget', '-O', file_path, file_url])  # Download the file using wget

        # Convert the PDF to images
        output_prefix = os.path.splitext(file_path)[0]
        subprocess.run(['pdftoppm', '-png', file_path, output_prefix])

        # Send the images one by one
        image_files = [f for f in os.listdir('.') if f.startswith(output_prefix)]
        for image_file in image_files:
            with open(image_file, 'rb') as img:
                bot.send_photo(message.chat.id, img)

        # Cleanup: Delete the PDF and image files
        os.remove(file_path)
        for image_file in image_files:
            os.remove(image_file)

    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your request.")
# Handler for photos
@bot.message_handler(content_types=['photo'])
def handle_photos(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        file_url = f'https://api.telegram.org/file/bot{TelegramBOT_TOKEN}/{file_info.file_path}'
        file_path = f"downloads/{os.path.basename(file_info.file_path)}"

        # Download the file
        file_response = requests.get(file_url)
        with open(file_path, 'wb') as f:
            f.write(file_response.content)

        # Convert image to PDF
        output_pdf_path = f"pdfs/{os.path.splitext(os.path.basename(file_path))[0]}.pdf"
        image_to_pdf([file_path], output_pdf_path)

        # Send the PDF
        with open(output_pdf_path, 'rb') as pdf:
            bot.send_document(message.chat.id, pdf)

    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your request.")



# Cat API Handler
@bot.message_handler(commands=['cat'])
def cat_handler(message):
    cat_image_url = get_random_cat_image()
    if cat_image_url:
        bot.send_photo(message.chat.id, cat_image_url)
    else:
        bot.reply_to(message, "Failed to fetch cat image.")

# PokeAPI Handler
@bot.message_handler(commands=['pokemon'])
def pokemon_handler(message):
    pokemon_name = message.text.split(maxsplit=1)[1]
    pokemon_info = get_pokemon_info(pokemon_name)
    bot.reply_to(message, pokemon_info)
# Handler for text messages
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        print(message)
        response = convo.send_message(message.text)
        bot.reply_to(message, response.text)
    except Exception as e:
        print(f"An error occurred: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your request.")

# Start polling for new messages
bot.polling(timeout=40)
