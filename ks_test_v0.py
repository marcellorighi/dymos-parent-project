import numpy as np 
import matplotlib.pyplot as plt


rho = 0.1
noise = np.random.normal(5, 450., 12)
noise[-1] = 200.
ks_noise = (1.0 / rho) * np.log(np.sum(np.exp(rho * noise)))

print(ks_noise)

plt.figure(figsize=(10, 3))
# plt.plot(np.exp(rho*noise), 'b-', marker='o', markersize=3)
plt.plot((noise), 'b-', marker='o', markersize=3)
plt.title('Airfoil Geometry Visualization')
plt.xlabel('x/c')
plt.ylabel('y/c')
plt.grid(True, alpha=0.3)
plt.show()

