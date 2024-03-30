import json
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Carregar dados do arquivo JSON
with open('dados.json', 'r') as file:
    data = json.load(file)

# Extrair textos e rótulos
texts = [entry['text'] for entry in data]
labels = np.array([entry['label'] for entry in data])

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Pipeline com TF-IDF e SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC())
])

# Definir grade de hiperparâmetros para busca
parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'svm__C': [0.1, 1, 10]
}

# Realizar busca em grade com validação cruzada
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Avaliar o melhor modelo no conjunto de teste
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Salvar o melhor modelo treinado
joblib.dump(best_model, 'modelo_IA.pkl')
print("Modelo treinado e salvo com sucesso!")
