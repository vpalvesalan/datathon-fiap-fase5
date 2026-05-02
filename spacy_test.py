import spacy

print("Carregando o modelo...")
# 1. Carrega o cartucho estatístico em português
nlp = spacy.load("pt_core_news_sm")

# 2. Uma frase de teste cheia de "dados sensíveis"
texto = "O analista Alan Turing acessou a conta do Banco Itaú em São Paulo no dia 15 de Maio."

# 3. O spaCy processa o texto (aplica as matrizes matemáticas)
doc = nlp(texto)

print("\n--- 1. ANÁLISE DE ENTIDADES (O foco do Presidio) ---")
print("O modelo procura nomes próprios, locais e organizações:\n")

for entidade in doc.ents:
    print(f"Encontrei: {entidade.text:<15} | Classificação: {entidade.label_}")

print("\n--- 2. ANÁLISE GRAMATICAL (Como ele lê a frase) ---")
print("O modelo entende a função de cada palavra:\n")

# Vamos imprimir apenas substantivos, verbos e nomes próprios para não poluir
for token in doc:
    if token.pos_ in ["VERB", "NOUN", "PROPN"]: 
        print(f"Palavra: {token.text:<15} | Tipo Gramatical: {token.pos_}")