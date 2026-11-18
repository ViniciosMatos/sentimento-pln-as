import re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

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

teste = "teste para trabalho GRêMIO GREMIO, não gosto de cholorado"
print("\nETAPA 2")
print("Original:", teste)
print("Completo minusculo:", preprocessar(teste))
print("Sem stopwords:", preprocessar(teste, remover_stopwords=True))

# ETAPA 3

X_treino, X_teste, y_treino, y_teste = train_test_split(
    df["texto"],
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

    textos_treino = [
        preprocessar(texto, remover_stopwords)
        for texto in X_treino
    ]
    textos_teste = [
        preprocessar(texto, remover_stopwords)
        for texto in X_teste
    ]

    vetorizador = Vetorizador()
    X_treino_vetorizado = vetorizador.fit_transform(textos_treino)
    X_teste_vetorizado = vetorizador.transform(textos_teste)

    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_treino_vetorizado, y_treino)

    print(
        f"{config}: stopwords {stopwords_config}, "
        f"vocabulário={len(vetorizador.get_feature_names_out())}"
    )

# ETAPA 4

print("ETAPA 4")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = []

for config, stopwords_config, Vetorizador in configs:
    remover_stopwords = stopwords_config == "removidas"
    textos_treino = [
        preprocessar(texto, remover_stopwords)
        for texto in X_treino
    ]
    sentimentos_treino = list(y_treino)
    acuracias = []
    f1s = []
    vocabulario = []

    for treino, validacao in cv.split(textos_treino, sentimentos_treino):
        textos_fold_treino = [textos_treino[i] for i in treino]
        textos_fold_validacao = [textos_treino[i] for i in validacao]
        sentimentos_fold_treino = [sentimentos_treino[i] for i in treino]
        sentimentos_fold_validacao = [sentimentos_treino[i] for i in validacao]

        vetorizador = Vetorizador()
        X_fold_treino = vetorizador.fit_transform(textos_fold_treino)
        X_fold_validacao = vetorizador.transform(textos_fold_validacao)
        vocabulario = vetorizador.get_feature_names_out()

        modelo = LogisticRegression(max_iter=1000, random_state=42)
        modelo.fit(X_fold_treino, sentimentos_fold_treino)
        previsoes = modelo.predict(X_fold_validacao)

        acuracias.append(
            accuracy_score(sentimentos_fold_validacao, previsoes)
        )
        f1s.append(
            f1_score(sentimentos_fold_validacao, previsoes, average="macro")
        )

    resultado = {}
    resultado["config"] = config
    resultado["stopwords"] = stopwords_config
    resultado["vetorizador"] = Vetorizador.__name__
    resultado["vocabulario"] = len(vocabulario)
    resultado["acc_cv_media"] = sum(acuracias) / len(acuracias)
    resultado["acc_cv_desvio"] = pd.Series(acuracias).std(ddof=0)
    resultado["f1_cv_media"] = sum(f1s) / len(f1s)
    resultado["f1_cv_desvio"] = pd.Series(f1s).std(ddof=0)
    resultados.append(resultado)

tabela = pd.DataFrame(resultados)
tabela = tabela.sort_values(
    ["acc_cv_media", "f1_cv_media"],
    ascending=False,
)

tabela.to_csv("resultados/comparacao.csv", index=False)

print(tabela.to_string(index=False))
print("Configuração escolhida:", tabela.iloc[0]["config"])
