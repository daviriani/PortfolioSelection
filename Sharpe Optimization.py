# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:33:59 2023

@author: vinig
"""

import pandas as pd
import cvxpy as cp
import numpy as np

# 1. Importar os dados históricos dos ativos
df = pd.read_excel('dados.xlsx', sheet_name='DWJ')
df.to_csv('dados.csv', index=False)
df = pd.read_csv('dados.csv')

# 2. Dividir os dados em duas janelas temporais
in_sample = df.loc[(df['Data'] >= '1994-01-01') & (df['Data'] < '1999-01-01')]
in_sample = in_sample.drop(columns='TLR')
in_sample = in_sample.fillna(0)

out_of_sample = df.loc[(df['Data'] >= '1999-01-01') & (df['Data'] < '2004-01-01')]
out_of_sample = out_of_sample.drop(columns='TLR')



# 3. Estimar os parâmetros do modelo de otimização utilizando a janela in sample
mu = in_sample.mean().values
mu[np.isnan(mu)] = 0

# 4. Matriz de Covariância da amostra
Sigma = in_sample.cov().values
Sigma[np.isnan(Sigma)]= 0

# 5. Número de ativos da carteira
n_assets = len(mu)

# 6. Criação da variável de peso dos ativos (w)
w = cp.Variable(n_assets)

#7. Definição da função de retorno esperado da carteira e matriz de covariância

ret = mu
risk = cp.quad_form(w, Sigma)

#8. Defina o parâmetro de regularização
lambda_ = cp.Parameter(nonneg=True)

#9. Defina a função de utilidade quadrática
utility = w.T @ ret - (lambda_ / 2)*risk

objective = cp.Maximize(utility)

constraints = [
    w >= 0,
    cp.sum(w) == 1
]

problem = cp.Problem(objective, constraints)

#10. Colocando os lambdas para percorrer o domínio
#lambdas = [0.01]
lambdas = [0.01, 0.1, 1, 10, 100]
#lambdas = [0.01, 0.1, 1, 10, 100]
#lambdas = [0.01, 0.1, 1, 10, 100]
#lambdas = [0.01, 0.1, 1, 10, 100]

for lam in lambdas:
    lambda_.value = lam
    problem.solve()
    print(f"Para lambda = {lam}, status = {problem.status}, retorno esperado = {ret.T @ w.value}, volatilidade = {np.sqrt(w.value.T @ Sigma @ w.value)}")


#11. Pegar a carteira ótima in sample
w.value

#12. Calcular o retorno in sample (otimizado dentro da amostra)
# 12.1. Tirando os valores de NaN:
in_sample = in_sample.fillna(0.00)

# 12.2. Tirando a coluna Data dos valores In Sample:
in_sample2 = in_sample.drop(columns='Data')

# Encontrando o valor ótimo da carteira (In Sample):
retorno_in = in_sample2 @ w.value
print(retorno_in.sum())


# 13. Aplicar os parâmetros estimados na janela out of sample
# 13.1. Tirando os valores de NaN:
out_of_sample = out_of_sample.fillna(0.00)

# 13.2. Tirando a coluna Data dos valores Out of Sample:
out_of_sample2 = out_of_sample.drop(columns='Data')

#14. Calculando o retorno out of sample:
retorno = out_of_sample2 @ w.value
print (retorno.sum()) 

#15. Exportar o vetor de pesos, o resultado in sample e o resultado out of sample
peso = w.value
peso_df = pd.DataFrame(columns=['peso','in-sample','out-of-sample'])
peso_df['peso']=peso
peso_df.index=out_of_sample2.columns
peso_df['in-sample'] = retorno_in.sum()
peso_df['out-of-sample'] = retorno.sum()

peso_df.to_excel('resultado_Sharpe_4.xlsx')




