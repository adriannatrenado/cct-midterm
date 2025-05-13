
import pandas as pd #step 1
import arviz as az #step 6
import matplotlib.pyplot as plt
import numpy as np
def load_plant_knowledge_data(filepath='cct-midterm/data/plant_knowledge.csv'):
    df = pd.read_csv(filepath)
    data_matrix = df.drop(columns=['Informant']).values #conv 2 numpy array
    return data_matrix

import pymc as pm #step 2
data = load_plant_knowledge_data()
N, M = data.shape #n informants and m items
with pm.Model() as cct_model:
    D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N) #prior di btwn 0.5 % 1
    Z = pm.Bernoulli("Z", p=0.5, shape=M)

    D_reshaped = D[:, None] #step 3 #reshape to n,1
    p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped) #cct formula

    X = pm.Bernoulli("X",p=p,observed=data) #step 4
    trace = pm.sample(draws=2000, chains=4, tune=1000, target_accept=0.9, return_inferencedata=True)

#step 6
summary = az.summary(trace, var_names=["D","Z"]) #summary table
print(summary)

az.plot_posterior(trace, var_names=["D"]) #posterior plot D
plt.savefig("d_posterior.png")
plt.clf()
az.plot_posterior(trace, var_names=["Z"]) #posterior plot Z
plt.savefig("z_posterior.png")
plt.clf()

D_means = trace.posterior["D"].mean(dim=("chain","draw")).values
print("\nposterior mean competence for each informant:\n", D_means)
most_competent = D_means.argmax()
least_competent = D_means.argmin()
print(f"\nmost competent informant is: {most_competent} (mean competence = {D_means[most_competent]:.2f})")
print(f"\nleast competent informant is: {least_competent} (least competence = {D_means[least_competent]:.2f})")

Z_means = trace.posterior["Z"].mean(dim=("chain", "draw")).values
Z_consensus = (Z_means > 0.5).astype(int)
print("\nposterior mean probability per consensus answer (Z):\n", Z_means)
print("\nconsensus answer key (rounded):\n", Z_consensus)

majority_vote = np.round(data.mean(axis=0)).astype(int)
print("\nmajority vote answer key:\n", majority_vote)
agreement = (Z_consensus == majority_vote).mean()
print(f"\nAgreement between CCT model and majority vote: {agreement:.2%}")

