import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler

# Download necessary NLP resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
data_path = "datasets/text_dataset/bot_detection_data.csv"
print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)
print(f"Dataset loaded successfully! Shape: {df.shape}")

# Ensure 'Tweet' column exists
df['Tweet'] = df['Tweet'].astype(str)

# Text Preprocessing Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+", '', text)  # Remove mentions
    text = re.sub(r"#", '', text)  # Remove hashtag symbol but keep text
    text = re.sub(r"\W", ' ', text)  # Remove special characters
    text = re.sub(r"\s+", ' ', text).strip()  # Remove extra spaces
    return text

df['cleaned_text'] = df['Tweet'].apply(clean_text)

# Remove stopwords and apply lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(
    [lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))

# Convert text into numerical features
vectorizer = TfidfVectorizer(max_features=1000, smooth_idf=True, sublinear_tf=True, norm='l2')
text_features = vectorizer.fit_transform(df['cleaned_text'])

# Balance dataset with oversampling
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(text_features, df['Bot Label'])

# Convert to DataFrame and add label back
processed_df = pd.DataFrame(X_resampled.toarray())
processed_df['Bot Label'] = y_resampled  

# ✅ Fix for "Input X must be non-negative" in Chi-square feature selection
processed_df.iloc[:, :-1] = abs(processed_df.iloc[:, :-1])  

# Save preprocessed data
processed_data_path = "datasets/text_dataset/processed_bot_data.csv"
processed_df.to_csv(processed_data_path, index=False)
print(f"Text preprocessing complete! Processed data saved at: {processed_data_path}")
