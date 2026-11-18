RELATORIO — Análise de Sentimento com PLN
Nome: Vinicios Magnus de Matos
Repositório: 
Data: 17/11/2026


Tabela-resumo (preencha com os seus resultados)
Config	Stopwords	Vetorizador	Vocabulário	Acurácia CV (média ± desvio)	F1 CV (média ± desvio)
A	mantidas	BoW			
B	mantidas	TF-IDF			
C	removidas	BoW			
D	removidas	TF-IDF			
Configuração escolhida: C Acurácia no teste: 0.85 F1 no teste: 0.85

1. Pré-processamento
Quais decisões você tomou (minúsculas, pontuação, números, stopwords)? Qual o tamanho do vocabulário com e sem remoção de stopwords? O que você fez com a palavra "não" e por quê?

A decisão foi não adicionar o 'não' como stopword, porque ele traz um significado importante à frase, e uma alteração pode mudar completamente o sentido da mesma. Por exemplo: "Não gosto do Internacional", se removermos, a frase ficaria oposta "gosto do internacional" (inaceitável)

2. BoW vs. TF-IDF
Compare as quatro configurações levando em conta o desvio padrão. As diferenças são relevantes? Qual você escolheu e com base em quê? Explique a diferença conceitual entre as duas representações.

A minha escolha seria a configuração C, com BoW, e "ativando" as stopwords. Os resultados de acurácia e F1 foram muito próximos, mas os desvios foram menores da Configuração C, mostrando que foi o modelo mais estável neste dataset.

3. Avaliação no teste
Mostre a matriz de confusão e interprete precisão, recall e F1 de cada classe. Para uma loja, qual erro é mais grave: classificar uma reclamação como elogio ou o contrário? Por que a acurácia é menos enganosa aqui do que no exemplo de fraude da Aula 8?

(sua resposta)

4. Validação cruzada vs. divisão única
Por que a validação cruzada é mais confiável do que um único conjunto de teste para comparar configurações? Por que a escolha não pode ser feita olhando o teste? O que seria vazamento de dados neste trabalho e como você o evitou?

(sua resposta)

5. Análise de erros e palavras influentes
Escolha 3 textos de erros.csv e explique por que o modelo pode ter errado. Observe as 10 palavras mais influentes de cada classe: há alguma que não carrega sentimento por si só (ex.: palavra funcional ou de domínio)? O que isso diz sobre o que o modelo realmente aprendeu?

(sua resposta)

6. Casos difíceis e limitações
Com base em casos_dificeis_predicoes.csv, como o modelo se saiu em negação, ironia e frases mistas? Explique por que BoW/TF-IDF têm essa limitação, proponha duas melhorias (cite técnicas vistas na Aula 11) e discuta duas limitações do dataset e um risco de usar este modelo em produção.

(sua resposta)

Declaração de uso de IA
Marque uma opção e complete:

 Não usei IA generativa em nenhuma parte deste trabalho.
 Usei IA generativa somente para: ______________ (não é permitido gerar código ou relatório).
