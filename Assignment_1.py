'''
Create a Flask WebApp which takes a news article URL form the user and does the following:

1. Extract the news text and clean it.

2. Analyse the text for number of sentences, words, POS tags, and any other relevant information. Display this on the website in a nicely formatted manner after the URL is submitted.

3. Store the URL, news text, and a summary of your analysis in a PostgreSQL table.

4. There should be a button on the website to view history of URLs submitted and the associated analysis done. This should only be visible and accessible by the website admin.


'''
from flask import Flask, render_template, request, redirect, url_for, session
import requests
from bs4 import BeautifulSoup
import nltk
import json
import re
from nltk import sent_tokenize, word_tokenize, pos_tag
nltk.download('all')
import psycopg2
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask import jsonify
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
from authlib.integrations.flask_client import OAuth
from flask import flash


app = Flask(__name__, static_folder="/var/data/")

oauth = OAuth(app)

app.config['SECRET_KEY'] = "THIS SHOULD BE SECRET"
app.config['GITHUB_CLIENT_ID'] = "4d659ffa6e7885ee063a"
app.config['GITHUB_CLIENT_SECRET'] = "c7ec0ef23d7a2a917e63c5dce07000e1920df101"

github = oauth.register(
    name='github',
    client_id=app.config["GITHUB_CLIENT_ID"],
    client_secret=app.config["GITHUB_CLIENT_SECRET"],
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)

# GitHub admin usernames for verification
github_admin_usernames = ["7773aakash", "atmabodha"]


# Establish a connection to the PostgreSQL database
conn = psycopg2.connect(host='dpg-cnmp1co21fec73986npg-a', database='url_datas', user='url_datas_user', password='lfiIGjhOT5iMJXU8ig5Isw7LApjBwnuE')
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS url_summary(
        url VARCHAR NOT NULL,
        text  TEXT,
        no_of_sentences INTEGER,
        stop_words INTEGER,
        upos_tags JSONB
    )
""")
conn.commit()


# Load stop words
stp_words = set(nltk.corpus.stopwords.words('english'))

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')




@app.route('/u', methods=['POST'])
def process_url():
    try:
        url = request.form['url']
        news_data = extract_and_clean_news_text(url)

        # Access the title, cleaned text, and analysis separately
        title = news_data['title']
        cleaned_text = news_data['cleaned_text']

        analysis = analyze_text(cleaned_text)

        # Generate word cloud
        wordcloud_img = generate_wordcloud(cleaned_text)
        sentiment_labels, sentiment_percentages = calculate_sentiment(cleaned_text)
        # Generate summary
        summary = generate_summary(cleaned_text, num_sentences=11)

        # # Insert data into the PostgreSQL database
        cur.execute(
            "INSERT INTO url_summary (url, text, no_of_sentences, stop_words, upos_tags) VALUES (%s, %s, %s, %s, %s)",
            (url, cleaned_text, analysis["no_of_sentences"], analysis["stop_word_count"],
             json.dumps(analysis['upos_tags'])))
        conn.commit()

        # Display the title and analysis on the website
        return render_template('front.html', url=url, title=title, cleaned_text=cleaned_text, analysis=analysis,
                               wordcloud_img=wordcloud_img, sentiment_labels=sentiment_labels,
                               sentiment_percentages=sentiment_percentages, summary=summary, error_message=None)

    except Exception as e:
        # Handle the exception and pass the error message to the template
        error_message = f"An error occurred: {str(e)}"
        return render_template('front.html', url=None, title=None, cleaned_text=None, analysis=None,
                               wordcloud_img=None, sentiment_labels=None,
                               sentiment_percentages=None, summary=None, error_message=error_message)


@app.route('/history_password', methods=['GET', 'POST'])
def history_password():
    if request.method == 'POST':
        password_attempt = request.form['password']

        correct_password = "@7773"  # Password to go histry page

        if password_attempt == correct_password:
            cur = conn.cursor()
            # Password is correct, render the history page
            cur.execute("select * from url_summary")
            history_data = cur.fetchall()
            return render_template('history.html', history_data=history_data)
        else:
            flash("Incorrect password. Please try again.", "error")

    # Render the password entry form
    return render_template('password_entry.html')
def extract_and_clean_news_text(url):
    # we use class method to extract title and cleaned text of news
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find and extract the title from h1 tag within the element with class HNMDR
    title_element = soup.find(class_='HNMDR')
    title = title_element.get_text() if title_element else ""

    # Extract and clean the text using BeautifulSoup
    main_text_element = soup.find(class_='_s30J clearfix')
    main_text = main_text_element.get_text() if main_text_element else ""
    
    # Remove HTML tags using regex 
    cleaned_text = re.sub(r'<.*?>', '', str(main_text))
    title = re.sub(r'<.*?>', '', str(title))
    
    # Replace periods at the end of sentences with period + space
    cleaned_text = re.sub(r'(?<=[.!?])\s*(?=[A-Z])', ' ', cleaned_text)
    title = re.sub(r'(?<=[.!?])\s*(?=[A-Z])', ' ', title)
    
    return {'cleaned_text': cleaned_text, 'title': title}

def analyze_text(cleaned_text):
    words = word_tokenize(cleaned_text)
    pos_tags = pos_tag(words, tagset='universal')
    
    upos_counts = {}
    for word, tag in pos_tags:
        if word.isalpha():
            if tag in upos_counts:
                upos_counts[tag] += 1
            else:
                upos_counts[tag] = 1

    stop_word_count = sum(1 for word in words if word.lower() in stp_words)

    word_count = len(words)
    no_of_sentences = len(sent_tokenize(cleaned_text))
    
    # Calculate estimated reading time in minutes
    average_reading_speed = 170  # Words per minute (adjust as needed)
    estimated_reading_time = round(word_count / average_reading_speed, 2)

    analysis = {
        'no_of_sentences': no_of_sentences,
        'word_count': word_count,
        'upos_tags': upos_counts,
        'stop_word_count': stop_word_count,
        'estimated_reading_time': estimated_reading_time
    }
    return analysis

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{img_data}"

def calculate_sentiment(text):
    sia = SentimentIntensityAnalyzer()

    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Initialize counters
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    # Analyze the sentiment of each sentence
    for sentence in sentences:
        sentiment_score = sia.polarity_scores(sentence)['compound']

        if sentiment_score >= 0.05:
            positive_count += 1
        elif sentiment_score <= -0.05:
            negative_count += 1
        else:
            neutral_count += 1

    # Calculate percentages
    total_sentences = len(sentences)
    positive_percentage = (positive_count / total_sentences) * 100
    negative_percentage = (negative_count / total_sentences) * 100
    neutral_percentage = (neutral_count / total_sentences) * 100

    # Return sentiment labels and percentages
    sentiment_labels = ['Positive', 'Negative', 'Neutral']
    sentiment_percentages = [positive_percentage, negative_percentage, neutral_percentage]

    return sentiment_labels, sentiment_percentages

def generate_summary(text, num_sentences=5):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # Filter out stop words
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    # Calculate word frequencies
    word_frequencies = FreqDist(filtered_words)

    # Calculate sentence scores based on word frequencies
    sentence_scores = {i: sum(word_frequencies[word] for word in word_tokenize(sent.lower()) if word.isalpha()) for i, sent in enumerate(sentences)}

    # Get the indices of top-scoring sentences
    top_sentences_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    # Sort selected sentences by their original order
    top_sentences_indices.sort()

    # Create the summary
    summary = ' '.join(sentences[i] for i in top_sentences_indices)

    return summary

github_admin_usernames = ["7773aakash", "atmabodha"]
@app.route('/login/github')
def github_login():
    github = oauth.create_client('github')
    redirect_uri = url_for('github_authorize', _external=True)
    return github.authorize_redirect(redirect_uri)

@app.route('/login/github/authorize')
def github_authorize():
    try:
        github = oauth.create_client('github')
        token = github.authorize_access_token()
        session['github_token'] = token
        resp = github.get('user').json()
        print(f"\n{resp}\n")
        logged_in_username = resp.get('login')
        if logged_in_username in github_admin_usernames:
            # Fetch data from the PostgreSQL database
            cur = conn.cursor()
            cur = conn.cursor()
            cur.execute("SELECT * FROM url_summary")
            history_data = cur.fetchall()
            # conn.close()
            return render_template('history.html', history_data=history_data)
        else:
            return redirect(url_for('index'))
    except:
        return redirect(url_for('index'))

@app.route('/logout/github')
def github_logout():
    session.clear()
    print("logout")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
