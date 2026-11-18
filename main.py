import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/Avaliacoes.csv")

numero_linhas = len(df)
numero_colunas = len(df.columns)
classes = df["sentimento"].value_counts()

# Conta as palavras de cada avaliação
df["num_palavras"] = df["texto"].str.split().str.len()
media_classe = df.groupby("sentimento")["num_palavras"].mean()

# Pega três exemplos de cada classe
ex_negativos = df[df["sentimento"] == "negativo"]["texto"].head(3)
ex_positivos = df[df["sentimento"] == "positivo"]["texto"].head(3)

print("Linhas:", numero_linhas)
print("Colunas:", numero_colunas)

print("\nCLASSES:")
for classe in classes.index:
    print(f"{classe}: {classes[classe]}")

print("\nTamanho médio dos textos:")
print("NEGATIVO:", media_classe['negativo'])
print("POSITIVO:", media_classe['positivo'])

print("\n3 exemplos negativos:")
for ex in ex_negativos:
    print(ex)

print("\n3 exemplos positivos:")
for ex in ex_positivos:
    print(ex)

stopwords = [
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "e", "é", "de", "do", "da", "dos", "das", "em",
    "no", "na", "nos", "nas", "para", "com", "por",
]

# ETAPA 2 ----

def preprocessar(texto, remover_stopwords=False):
    texto = texto.lower()
    palavras = re.findall(r"[a-záàâãéêíóôõúüç]+", texto)

    if remover_stopwords:
        palavras = [palavra for palavra in palavras if palavra not in stopwords]

    return " ".join(palavras)

# teste = "teste para trabalho GRêMIO GREMIO, não gosto de cholorado"
# print("\nETAPA 2")
# print("Original:", teste)
# print("Completo minusculo:", preprocessar(teste))
# print("Sem stopwords:", preprocessar(teste, remover_stopwords=True))

# ETAPA 3

# limpando os textos
limpos = [preprocessar(texto) for texto in df["texto"]]

# treinando o modelo
X_treino, X_teste, y_treino, y_teste = train_test_split(
    limpos,
    df["sentimento"],
    test_size=0.25,
    stratify=df["sentimento"],
    random_state=42,
)

configs = [
    ("A", "mantidas", CountVectorizer),
    ("B", "mantidas", TfidfVectorizer),
    ("C", "removidas", CountVectorizer),
    ("D", "removidas", TfidfVectorizer),
]

print("\nETAPA 3")

for config, stopwords_config, Vetorizador in configs:
    remover_stopwords = stopwords_config == "removidas"

    limpos_treino = [
        preprocessar(texto, remover_stopwords)
        for texto in X_treino
    ]
    limpos_teste = [
        preprocessar(texto, remover_stopwords)
        for texto in X_teste
    ]

    vetorizador = Vetorizador()
    X_treino_vetorizado = vetorizador.fit_transform(limpos_treino)
    X_teste_vetorizado = vetorizador.transform(limpos_teste)

    vocabulario = vetorizador.get_feature_names_out()

    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_treino_vetorizado, y_treino)

    print(
        f"{config}: stopwords {stopwords_config} vocabulário={len(vocabulario)}"
    )
    print(list(vocabulario))
