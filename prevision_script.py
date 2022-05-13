import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# importar base de dados para o python
tabela = pd.read_csv("advertising.csv")

# =====> Análise Exploratória: vamos entender como a base de dados está se comportando
# - Vamos ver a correlação entre cada um dos itens

# 1- Cálculo de correlação entre duas propriedades
# Cria o gráfico
sns.heatmap(tabela.corr(), annot=True, cmap="Wistia") # Heatmap criado com o seaborn, biblioteca criada com o matplotlib

# Exibe o gráfico
plt.show()

# Em previsão, X são as variáveis e Y é quem você quer prever
x = tabela[["TV", "Radio", "Jornal"]]
y = tabela["Vendas"]

# Separação das porcentagens de treino da IA
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.1)

# =====> Temos um problema de regressão - Vamos escolher os modelos que vamos usar entre:
# - Regressão Linear
# - RandomForest (Árvore de Decisão)

# ===> Treinamento dos modelos
# Regressão Linear
modelo_regressaolinear = LinearRegression()
modelo_regressaolinear.fit(x_treino, y_treino)

# Árvore de Decisão
modelo_arvoredecisao = RandomForestRegressor()
modelo_arvoredecisao.fit(x_treino, y_treino)


# =====> Teste da AI e Avaliação do Melhor Modelo
# - Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

print(f"{'Regressão Linear':>17}: {r2_score(y_teste, previsao_regressaolinear)}")
print(f"{'Árvore de Decisão':>17}: {r2_score(y_teste, previsao_arvoredecisao)}")
# O modelo com melhor resultado é a árvore de decisão


# =====> Visualização Gráfica das Previsões
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['Previsões Regressão Linear'] = previsao_regressaolinear
tabela_auxiliar['Previsões Árvore de Decisão'] = previsao_arvoredecisao

plt.figure(figsize=(15, 6)) # tamanho do gráfico
sns.lineplot(data=tabela_auxiliar) # criação do gráfico
plt.show() # display do gráfico


# =====> Fazer a nova previsão
novos = pd.read_csv('novos.csv')
previsao = modelo_arvoredecisao.predict(novos)
novos["Previsão de Lucro (Milhões)"] = previsao

print(novos)

