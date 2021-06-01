import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed()
m= 100  #cree 100 exemples
X = np.linspace(0,10, m).reshape(m,1)
y = X+ np.random.randn(m,1)

from  sklearn.linear_model import LinearRegression
model_1 = LinearRegression()
model_1.fit(X, y)
model_1.score(X, y)
prediction_1 = model_1.predict(X)
model_1.predict(X)
plt.scatter(X,y)
