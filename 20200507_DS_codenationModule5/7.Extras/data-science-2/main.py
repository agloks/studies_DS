#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm


# In[2]:


# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

# sns.set()


# In[3]:


athletes = pd.read_csv("athletes.csv")


# In[4]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
athletes.head(6)


# In[6]:


_sample_height = get_sample(athletes, ["height"], 3000)  
athletes.describe()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[7]:


def q1():
    # Retorne aqui o resultado da questão 1.  
    return sct.shapiro(_sample_height)[1] > 0.05

q1()


# In[8]:


print(f"Q1 result == {q1()}\n")
help(sct.shapiro)

sns.distplot(_sample_height, bins=25)


# In[9]:


help(sm.qqplot)
sm.qqplot(_sample_height, line="45", fit=True)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[24]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return bool(sct.jarque_bera(_sample_height)[1] > 0.05)
    
q2()


# In[11]:


print(f"Q2 result == {q2()}\n")
help(sct.jarque_bera)


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[12]:


_sample_weight = get_sample(athletes, "weight", 3000)

def q3():
    # Retorne aqui o resultado da questão 3.
    return bool(sct.stats.normaltest(_sample_weight)[1] > 0.05)
    
q3()


# In[13]:


help(sct.stats.normaltest)
sns.distplot(_sample_weight, bins=25)


# In[14]:


sns.boxplot(_sample_weight)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[15]:


_sample_weight_log = np.log(_sample_weight)

def q4():
    # Retorne aqui o resultado da questão 4.
    return bool(sct.stats.normaltest(_sample_weight_log)[1] > 0.05)

q4()


# In[16]:


print(sct.stats.normaltest(_sample_weight_log))
sns.distplot(_sample_weight_log, bins=25)
# sns.distplot(_sample_weight, bins=25)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[17]:


bra = athletes[athletes["nationality"] == "BRA"]["height"]
usa = athletes[athletes["nationality"] == "USA"]["height"]
can = athletes[athletes["nationality"] == "CAN"]["height"]

print(f"bra_na = {bra.isna().sum()}\t usa_na = {usa.isna().sum()}\t can_na = {can.isna().sum()}")
bra.dropna(inplace=True)
usa.dropna(inplace=True)
can.dropna(inplace=True)
print(f"bra_na = {bra.isna().sum()}\t usa_na = {usa.isna().sum()}\t can_na = {can.isna().sum()}")


# In[34]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return bool(sct.stats.ttest_ind(bra, usa, nan_policy="omit", equal_var=False)[1] > 0.05)

q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[33]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return bool(sct.stats.ttest_ind(bra, can, nan_policy="omit", equal_var=False)[1] > 0.05)

q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[32]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return float(sct.stats.ttest_ind(usa, can, nan_policy="omit", equal_var=False)[1].round(8))

q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# In[21]:


ax_bra = sns.distplot(bra, hist=False, kde_kws={"legend":True, "label":"BRA"})
ax_usa = sns.distplot(usa, hist=False, kde_kws={"legend":True, "label":"USA"})
ax_can = sns.distplot(can, hist=False, kde_kws={"legend":True, "label":"CAN"})

data_lines_bra = {"x": ax_bra.lines[0].get_xdata(), "y": ax_bra.lines[0].get_ydata()} 
data_lines_usa = {"x": ax_usa.lines[0].get_xdata(), "y": ax_usa.lines[0].get_ydata()}
data_lines_can = {"x": ax_can.lines[0].get_xdata(), "y": ax_can.lines[0].get_ydata()} 

# plt.axvline(data_lines_bra["x"][np.argmax(data_lines_bra["y"])], color='blue')
# plt.axvline(data_lines_usa["x"][np.argmax(data_lines_usa["y"])], color='red')
# plt.axvline(data_lines_can["x"][np.mean(data_lines_can["y"])], color='green')
print(np.mean(data_lines_bra["y"]))
print(np.mean(data_lines_usa["y"]))
print(np.mean(data_lines_can["y"]))
# np.argmax(data_lines_can["y"])


# In[22]:


sns.boxplot(data = [bra, usa, can])

