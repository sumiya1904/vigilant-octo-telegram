import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
data = {
    'text': [
        'Bonjour, comment ça va ?',
        'Hello, how are you?',
        'Hola, ¿cómo estás?',
        'Hallo, wie geht es dir?',
        'السلام عليكم',
        'Ciao, come stai?',
        'привет, как дела?',
        '你好，你怎么样？'
    ],
    'language': ['French', 'English', 'Spanish', 'German', 'Arabic', 'Italian', 'Russian', 'Chinese']
}

dataframe = pd.DataFrame(data)

# 2. Text Preprocessing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataframe['text'])
y = dataframe['language']

# 3. Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. Test with New Sentences
def detect_language(sentence):
    vec_sentence = vectorizer.transform([sentence])
    return model.predict(vec_sentence)[0]

new_sentences = [
    'Wie geht es dir?',  # German
    'Bonjour tout le monde',  # French
    '¿Cómo están todos?',  # Spanish
    'حالك كيف انت؟'  # Arabic
]

for sentence in new_sentences:
    print(f"'{sentence}' is detected as: {detect_language(sentence)}")
