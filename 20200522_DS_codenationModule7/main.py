#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

# sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


#Transforming object to float64
#Good practice? NEVER, but i was enjoying do this.
countries = countries.apply(lambda x: x if (str(x) == "nan" and type(x[0]) != str) else x.str.replace(",", ".").astype(float) if (type(x[0]) == str) and (len(str(x.str.extract("(?P<RESULT>[a-zA-z]+)").RESULT[0])) < 4) else x)

# countries = countries.apply(\
#                 lambda x: \
#                 x \
#                 if (str(x) == "nan" and type(x[0]) != str) \
#                 else x.str.replace(",", ".").astype(float) \
#                 if (type(x[0]) == str) and \
#                     (len(str(x.str.extract("(?P<RESULT>[a-zA-z]+)").RESULT[0])) < 4)\
#                 else x
#                )


# In[6]:


def applyEachElement(df, function, columns):
    new_df = df
    
    for column in columns:
        new_df[column] = new_df[column].apply(function)
    
    return new_df


# In[7]:


#Removing extra spaces in the columns Country and Region
columnsNeedStrip = ["Country", "Region"]
countries = applyEachElement(countries, str.strip, columnsNeedStrip)


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[8]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return list(countries["Region"].sort_values().unique())

q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[9]:


'''
countries[["Pop_density"]] is the same that np.reshape(list(countries["Pop_density"]), (-1, 1))
'''

def q2():
    # Retorne aqui o resultado da questão 2.
    reshape_pop_density = np.reshape(list(countries["Pop_density"]), (-1, 1))
    K = KBinsDiscretizer(n_bins=10, 
                         encode="ordinal",
                         strategy="quantile").fit_transform(reshape_pop_density)
    quantiles_above_90 = [x for x in K if x >= 9]

    return len(quantiles_above_90)

q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[10]:


def q3():
    # Retorne aqui o resultado da questão 3.
    features_in_climate = OneHotEncoder()                           .fit(countries[["Climate", "Region"]]                           .fillna("0")                           .astype("str"))
    
    return len(features_in_climate.get_feature_names())

q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[11]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]

countries_train_index = countries.dtypes[(countries.dtypes == np.float64) | (countries.dtypes == np.int64)].index
countries_train = countries[countries_train_index]

test_country = pd.DataFrame(dict(zip(countries_train.columns, test_country[2:])), index=[0])


# In[12]:


def q4():
    # Retorne aqui o resultado da questão 4.
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    transformer = ColumnTransformer(
        [
            ("numbers", pipe, countries_train_index),
        ]
    )

    transformer.fit(countries_train)
    result = dict(zip(test_country.columns, transformer.transform(test_country)[0]))
    
    return float(result["Arable"].round(3))

q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[13]:


#Net_migration = imigrations - emigrations

def q5():
    # Retorne aqui o resultado da questão 4.
    quartiles = countries["Net_migration"].quantile([.25, .75]).to_list() 
    IQR = quartiles[1] - quartiles[0]
    outliers_baixo = quartiles[0] - (1.5 * IQR)
    outliers_acima = quartiles[1] + (1.5 * IQR)
    removeria = False
    
    return (len(countries[countries["Net_migration"] < outliers_baixo]["Net_migration"]),
            len(countries[countries["Net_migration"] > outliers_acima]["Net_migration"]),
            removeria)
    
    return tuple((int(outliers_lower),
                  int(outliers_upper),
                  False))

q5()


# In[14]:


sns.boxplot(countries["Net_migration"])
# pd.DataFrame.boxplot(countries["Net_migration"])


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[15]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", download_if_missing=False, categories=categories, shuffle=True, random_state=42)


# In[16]:


def q6():
    # Retorne aqui o resultado da questão 4.
    CV = CountVectorizer()
    result = CV.fit_transform(newsgroup.data)
    
    return int(result                .getcol(CV.vocabulary_["phone"])                .sum())

q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[17]:


def q7():
    # Retorne aqui o resultado da questão 4.
    TFID = TfidfVectorizer(use_idf=True)
    result = TFID.fit_transform(newsgroup.data)
    
    return float(result                  .getcol(TFID.vocabulary_["phone"])                  .sum()                  .round(3))

q7()

