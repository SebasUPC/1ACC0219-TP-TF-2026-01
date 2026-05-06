import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 1. CARGA DE DATOS (Simulada para el ejemplo)
# Para el Hito 1, aquí cargarías tus archivos .txt o .csv con los cuentos
data = {
    'titulo': ['El Aleph', 'El Inmortal', 'Las Ruinas Circulares', 'Funes el Memorioso'],
    'texto': [
        'El diámetro del Aleph sería de dos o tres centímetros, pero el espacio cósmico estaba ahí...',
        'Salí de la ciudad de los inmortales y crucé el desierto para encontrar el fin de mi camino...',
        'Nadie lo vio desembarcar en la unánime noche, nadie vio la canoa de bambú hundiéndose...',
        'Ireneo Funes recordaba cada hoja de cada árbol de cada monte, cada una de las veces que la había visto...'
    ],
    'tema': ['Infinito', 'Identidad', 'Metafísica', 'Identidad'] # Etiquetas para clasificación
}

df = pd.DataFrame(data)

# 2. PREPROCESAMIENTO (Limpieza de texto)
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

def clean_text(text):
    tokens = word_tokenize(text.lower())
    # Eliminar puntuación y stopwords
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(words)

df['texto_limpio'] = df['texto'].apply(clean_text)

# 3. ANÁLISIS EXPLORATORIO (EDA)
print("--- Estadísticas Básicas ---")
df['word_count'] = df['texto'].apply(lambda x: len(x.split()))
print(df[['titulo', 'word_count']])

# Visualización 1: Distribución de temas
plt.figure(figsize=(8, 4))
sns.countplot(x='tema', data=df, palette='viridis')
plt.title('Distribución de Temas en el Dataset')
plt.show()

# Visualización 2: Nube de Palabras
all_words = " ".join(df['texto_limpio'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# 4. MODELIZACIÓN (Baseline - Hito 1)
# Vectorización por TF-IDF
tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(df['texto_limpio']).toarray()
y = df['tema']

# División para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Base: SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

print("\n--- Resultados Preliminares del Modelo ---")
# Nota: Con 4 ejemplos fallará, necesitas llenar el dataset para que funcione
# print(classification_report(y_test, model.predict(X_test)))
