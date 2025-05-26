import bayespy as bp
import numpy as np

# Simplified dataset: [Age, Cholesterol, HeartDisease]
# 0: young, 1: old | 0: normal, 1: high | 0: no, 1: yes
data = np.array([
    [0, 1, 1],
    [1, 1, 1],
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 1],
    [0, 1, 1]
])
N = len(data)

# Priors
p_age = bp.nodes.Dirichlet(np.ones(2))
age = bp.nodes.Categorical(p_age, plates=(N,))
age.observe(data[:, 0])

p_chol = bp.nodes.Dirichlet(np.ones(2))
chol = bp.nodes.Categorical(p_chol, plates=(N,))
chol.observe(data[:, 1])

# Heart disease depends on age and cholesterol
p_hd = bp.nodes.Dirichlet(np.ones(2), plates=(2, 2))  # (age, chol)
heart = bp.nodes.MultiMixture([age, chol], bp.nodes.Categorical, p_hd)
heart.observe(data[:, 2])

# Inference
p_hd.update()

# Predict for: old (1), high cholesterol (1)
test = bp.nodes.MultiMixture([1, 1], bp.nodes.Categorical, p_hd)
print("Probability of heart disease:", test.get_moments()[0][1])
