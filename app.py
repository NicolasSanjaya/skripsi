import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file, send_from_directory
import pandas as pd
from werkzeug.utils import secure_filename
import math
import pandas as pd
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
from livereload import Server
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ROWS_PER_PAGE = 10  # Number of rows to display per page

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables to store data and model
model = joblib.load('models/adaboost.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
df = None
hasil = None
preview_data = None
data_review = None

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def processing_data(df):
    # Cek apabila kolom full_text tidak ada
    if df['full_text'].isnull().all():
        flash('Kolom full_text tidak ditemukan')
        return redirect(url_for('processing'))

    # Preprocessing
    df = df.dropna(subset='full_text')
    df = df.drop_duplicates(subset='full_text')

    # 1. Mengambil kolom yang dibutuhkan
    dropcolumns = ['conversation_id_str', 'created_at', 'favorite_count', 'id_str', 'image_url', 'in_reply_to_screen_name', 'lang', 'location', 'quote_count', 'reply_count', 'retweet_count', 'tweet_url', 'user_id_str', 'username']
    df = df.drop(columns=dropcolumns, errors='ignore')


    # 2. Case folding - mengubah teks menjadi lowercase
    df['full_text'] = df['full_text'].str.lower()

    # 3. Cleaning - menghapus URL, mention, hashtag, dan karakter non alfanumerik
    def clean_text(text):
        # Menghapus URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Menghapus mention
        text = re.sub(r'@\w+', '', text)
        # Menghapus hashtag
        text = re.sub(r'#\w+', '', text)
        # Menghapus RT dan FAV
        text = re.sub(r'\brt\b', '', text)
        # Menghapus karakter non alfanumerik kecuali spasi
        text = re.sub(r'[^\w\s]', '', text)
        # Menghapus angka
        text = re.sub(r'\d+', '', text)
        # Menghapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        # Mengganti &amp dengan kata dan
        text = text.replace('&amp', 'dan')
        return text

    df['full_text'] = df['full_text'].apply(clean_text)

    # 4. Normalization
    # Fungsi untuk normalisasi kata (mengubah kata tidak baku ke baku)
    slang_df = pd.read_csv("slang.csv")
    slang_dict = dict(zip(slang_df['slang'], slang_df['formal']))

    def normalize_text(text, slang_dict):
        # Tokenisasi kata dan normalisasi
        def replace_slang(word):
            return slang_dict.get(word.lower(), word)

        # Menggunakan regex untuk mempertahankan tanda baca
        words = re.findall(r'\b\w+\b|\S', text)
        normalized_words = [replace_slang(word) for word in words]
        return ' '.join(normalized_words)

    df["stemmed_text"] = df["full_text"].apply(lambda x: normalize_text(x, slang_dict))

    # 5. Tokenizing
    df['stemmed_text'] = df['stemmed_text'].apply(word_tokenize)

    # 6. Stopword removal
    nltk.download('stopwords', quiet=True)
    # Ambil daftar stopword bahasa Indonesia
    stop_words = set(stopwords.words('indonesian'))

    def remove_stopwords(tokens):
        # Filter token yang bukan stopword
        return [word for word in tokens if word not in stop_words and len(word) > 2]

    df['stemmed_text'] = df['stemmed_text'].apply(remove_stopwords)

    # 7. Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stem_tokens(tokens):
        return [stemmer.stem(token) for token in tokens]

    df['stemmed_text'] = df['stemmed_text'].apply(stem_tokens)
    df['stemmed_text'] = df['stemmed_text'].apply(lambda x: ' '.join(x))

    df.to_csv('data.csv', index=False)
    
    return df


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download_excel', methods=['GET', 'POST'])
def download_excel():
    try:
        if os.path.exists("hasil.xlsx"):
            os.remove("hasil.xlsx")

        # Read the CSV file
        df = pd.read_csv('hasilweb.csv')

        # Save the DataFrame to an Excel file
        output_file_excel = 'hasil.xlsx'
        df.to_excel(output_file_excel, index=False)

        # Send the file for download
        return send_file(output_file_excel, as_attachment=True)
    except Exception as e:
        flash(f'Error generating Excel file: {str(e)}')
        return redirect(url_for('processing'))

@app.route('/download_csv', methods=['GET', 'POST'])
def download_csv():
    try:
        if os.path.exists("hasil.csv"):
            os.remove("hasil.csv")

        # Read the CSV file
        df = pd.read_csv('hasilweb.csv')

        # Save the DataFrame to an Excel file
        output_file_excel = 'hasil.csv'
        df.to_csv(output_file_excel, index=False)

        # Send the file for download
        return send_file(output_file_excel, as_attachment=True)
    except Exception as e:
        flash(f'Error generating Excel file: {str(e)}')
        return redirect(url_for('processing')) 
    
@app.route('/reset_data', methods=['GET','POST'])
def reset_data():
    global preview_data, hasil, data_review
    # Hapus file upload jika ada
    if 'uploaded_file' in session:
        session.pop('uploaded_file', None)
    if os.path.exists('data.csv'):
        os.remove('data.csv')
    if os.path.exists('hasilweb.csv'):
        os.remove('hasilweb.csv')
    preview_data = None
    hasil = None
    data_review = None
    flash('Data telah direset')

    return redirect(url_for('kelola_data'))

@app.route('/kelola_data', methods=['GET', 'POST'])
def kelola_data():
    preview_data = None
    total_rows = 0
    total_columns = 0
    current_page = request.args.get('page', 1, type=int)
    total_pages = 1

    
    # Check if a file was previously uploaded
    if 'uploaded_file' in session:
        try:
            df = pd.read_csv(session['uploaded_file'])
            total_rows, total_columns = df.shape
            total_pages = math.ceil(total_rows / ROWS_PER_PAGE)
            
            # Make sure current page is valid
            if current_page < 1:
                current_page = 1
            elif current_page > total_pages:
                current_page = total_pages
                
            # Calculate start and end indices for pagination
            start_idx = (current_page - 1) * ROWS_PER_PAGE
            end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
            
            # Get slice of dataframe for current page
            preview_data = df.iloc[start_idx:end_idx].to_html(classes='table table-striped')
        except Exception as e:
            flash(f'Error reading CSV file: {str(e)}')
            return redirect(request.url)
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Save the filename to the session
            session['uploaded_file'] = filepath
            
            # Read CSV for preview
            try:
                df = pd.read_csv(filepath)
                total_rows, total_columns = df.shape
                total_pages = math.ceil(total_rows / ROWS_PER_PAGE)
                current_page = 1
                
                # Get first page of data
                preview_data = df.iloc[0:ROWS_PER_PAGE].to_html(classes='table table-striped')
                flash('File successfully uploaded')
            except Exception as e:
                flash(f'Error reading CSV file: {str(e)}')
                return redirect(request.url)

    return render_template('kelola_data.html', 
                           preview_data=preview_data,
                           current_page=current_page,
                           total_pages=total_pages,
                           total_rows=total_rows,
                           total_columns=total_columns)

@app.route('/processing')
def processing():
    global preview_data, hasil
    total_rows = 0
    total_columns = 0
    current_page = request.args.get('page', 1, type=int)
    total_pages = 1

    if(os.path.exists("hasilweb.csv")):
        hasil = True

    # Check if a file has been uploaded in this session
    if 'uploaded_file' in session:
        try:
            # Read the CSV file for preview
            df = pd.read_csv(session['uploaded_file'])

            total_rows, total_columns = df.shape
            total_pages = math.ceil(total_rows / ROWS_PER_PAGE)
            
            # Make sure current page is valid
            if current_page < 1:
                current_page = 1
            elif current_page > total_pages:
                current_page = total_pages
                
            # Calculate start and end indices for pagination
            start_idx = (current_page - 1) * ROWS_PER_PAGE
            end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
            
            # Get slice of dataframe for current page
            preview_data = df.iloc[start_idx:end_idx].to_html(classes='table table-striped')
        except Exception as e:
            flash(f'Error reading CSV: {str(e)}')

    
    if data_review is not None or os.path.exists("hasilweb.csv"):
        df = pd.read_csv("hasilweb.csv")

        total_rows, total_columns = df.shape
        total_pages = math.ceil(total_rows / ROWS_PER_PAGE)
            
        # Make sure current page is valid
        if current_page < 1:
            current_page = 1
        elif current_page > total_pages:
            current_page = total_pages
                
        # Calculate start and end indices for pagination
        start_idx = (current_page - 1) * ROWS_PER_PAGE
        end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
            
        # Get slice of dataframe for current page
        preview_data = df.iloc[start_idx:end_idx].to_html(classes='table table-striped')

    return render_template('processing.html', preview_data=preview_data, 
                    hasil=hasil, current_page=current_page, total_pages=total_pages, total_rows=total_rows, total_columns=total_columns)

@app.route('/process_data', methods=['POST'])
def process_data():
    global data_review, preview_data
    
    # Menghapus File Yang Sudah Ada
    # File kolom full_text dan stemmed_text
    if(os.path.exists("data.csv")):
        os.remove("data.csv")
    # File kolom full_text, stemmed_text, dan sentiment
    if(os.path.exists("hasilweb.csv")):
        os.remove("hasilweb.csv")

    if  'uploaded_file' not in session:
        flash('Tidak ada file yang diunggah')
        return redirect(url_for('kelola_data'))
    
    data = pd.read_csv(session['uploaded_file'])

    # Preprocess the data
    processing_data(data)

    # Data yang belum ada kolom sentiment
    df = pd.read_csv("data.csv")

    # Sementara untuk testing (data yang sudah dipreprocessing) yang sudah ada kolom sentiment
    # data = pd.read_csv('hasilweb.csv')
    df['stemmed_text'] = df['stemmed_text'].dropna()
    df['stemmed_text'] = df['stemmed_text'].astype(str)
        
    X = df['stemmed_text']

    y_pred = model.predict(vectorizer.transform(X))

    df['predicted_sentiment'] = y_pred
    df['predicted_sentiment'] = df['predicted_sentiment'].astype(str)

    hasil = True

    df.drop(columns=['stemmed_text'], inplace=True, errors='ignore')

    df.to_csv('hasilweb.csv', index=False)

    data_review = df.to_html(classes='table table-striped')

    preview_data = data_review

    return redirect(url_for('processing'))

@app.route('/uji_coba', methods=['GET', 'POST'])
def uji_coba():
    prediction = None
    text = None
    if request.method == 'POST':
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model belum dilatih'})
    
        text = request.form.get('text', '')
        if not text:
            return jsonify({'error': 'Teks tidak boleh kosong'})
        
        # Vectorize the text
        text_vec = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_vec)[0]

    return render_template('uji_coba.html', result=prediction, text=text)

@app.route('/info_model')
def info_model():
    return render_template('info_model.html')



if __name__ == '__main__':

    server = Server(app.wsgi_app)

    app.run(debug=True)