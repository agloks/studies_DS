#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# In[3]:


print(f"Rows: {black_friday.shape[0]} | Columns: {black_friday.shape[1]}")
black_friday.head(6)


# ## Inicie sua análise a partir daqui

# In[4]:


black_friday.describe()


# In[5]:


black_friday.isna().sum()


# In[6]:


black_friday.dtypes


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[7]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[8]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return int(black_friday[(black_friday["Age"] == "26-35") & (black_friday["Gender"] == "F")]["Age"].count())


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[9]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday["User_ID"].nunique()
    


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[10]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[11]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return float(black_friday["Product_Category_3"].isna().sum()/black_friday.shape[0])


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[12]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return sorted([x for x in black_friday.isnull().sum()], reverse=True)[0]
    


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[13]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return black_friday["Product_Category_3"].dropna().mode()[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[14]:


def q8():
    # Retorne aqui o resultado da questão 8.
    return float((MinMaxScaler().fit_transform(np.array(black_friday["Purchase"]).reshape((-1,1)))).mean())
    


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[15]:


def q9():
    # Retorne aqui o resultado da questão 9.
    return len([x for x in (StandardScaler().fit_transform(np.array(black_friday["Purchase"]).reshape((-1,1)))) if x < 1 and x > -1])


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[16]:


def q10():
    # Retorne aqui o resultado da questão 10.
    return bool(len([k for k in range(0, len(black_friday["Product_Category_3"])) if (str(black_friday["Product_Category_2"][k]) == "nan" and str(black_friday["Product_Category_3"][k]) == "nan")]) == black_friday["Product_Category_2"].isna().sum()) 
    

