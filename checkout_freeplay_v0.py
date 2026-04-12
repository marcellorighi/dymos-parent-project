import numpy as np
import matplotlib.pyplot as plt 

beta =  15.0
delta = 0.25
x1 = np.linspace(-2.0,2.0, 1000) 
stiffness = x1 * (0.5 * np.tanh(beta * (x1 - delta)) + 0.5 * np.tanh(beta * (-x1 - delta)) + 1)

fig, ax = plt.subplots(figsize=(12, 9))

plt.plot(x1,stiffness) 

plt.show() 

