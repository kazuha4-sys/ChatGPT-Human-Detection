import joblib

# Carregando o modelo treinado
modelo_IA = joblib.load('modelo_IA.pkl')

# Exemplo de classificação de um novo texto
new_text = ""
prediction = modelo_IA.predict([new_text])[0]
if prediction == 1:
    print("O texto foi gerado pelo ChatGPT.")
else:
    print("O texto foi escrito por um humano.")