import re
import pandas as pd





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