# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:35:05 2023

@author: davir
"""

import cvxpy as cp
import numpy as np
import pandas as pd

# Carregar dados
df = pd.read_excel('dados.xlsx', sheet_name='DWJ')
df.to_csv('dados.csv', index=False)
df = pd.read_csv('dados.csv')

# Dividir os dados em duas janelas temporais
in_sample = df.loc[(df['Data'] >= '1994-01-01') & (df['Data'] < '1999-01-01')]
in_sample = in_sample.drop(columns='TLR')
in_sample = in_sample.fillna(0)

out_of_sample = df.loc[(df['Data'] >= '1999-01-01') & (df['Data'] < '2004-01-01')]
out_of_sample = out_of_sample.drop(columns='TLR')

# Estimar os parâmetros do modelo de otimização utilizando a janela in sample
mu = in_sample.mean().values
mu[np.isnan(mu)] = 0

# Matriz de Covariância da amostra
Sigma = in_sample.cov().values
Sigma[np.isnan(Sigma)] = 0

# Número de ativos da carteira
n_assets = len(mu)

# Criação da variável de peso dos ativos (w)
w = cp.Variable(n_assets)

# Cálculo do retorno ponderado pelo peso
weighted_returns = cp.matmul(in_sample.drop(columns='Data').values, w)

# Cálculo das probabilidades cumulativas
positive_returns = cp.maximum(weighted_returns, 0)
negative_returns = cp.minimum(weighted_returns, 0)
positive_prob = cp.sum(positive_returns) / cp.sum(in_sample.drop(columns='Data').values)
negative_prob = cp.sum(negative_returns) / cp.sum(in_sample.drop(columns='Data').values)

# Cálculo do Ômega Ratio
omega_ratio = positive_prob / negative_prob * cp.sum(positive_returns)

# Função objetivo: maximizar o Ômega Ratio
objective = cp.Maximize(omega_ratio)

# Restrições
constraints = [
    w >= 0,
    cp.sum(w) == 1
]

# Problema de otimização
problem = cp.Problem(objective, constraints)

# Solução do problema
problem.solve()

# Resultados da otimização
optimized_weights = w.value
optimized_omega = omega_ratio.value

print("Pesos otimizados:", optimized_weights)
print("Ômega Ratio otimizado:", optimized_omega)
