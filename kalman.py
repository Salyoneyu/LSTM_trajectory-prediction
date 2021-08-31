from numpy import ma
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

ship_data = np.array([
[114.27984,31.96803],
[114.27916,31.96783],
[114.28038,31.96796],
[114.28,31.96826],
[114.27974,31.96804],
[114.27954,31.96772],
[114.27964,31.96805],
[114.27974,31.9684	],
[114.28002,31.96783],
[114.27958,31.96807],
[114.27936,31.968],
[114.27962,31.96773],
[114.2798,31.968],
[114.2799,31.96796],
[114.27968,31.96799],
[114.27948,31.96771],
[114.2796,31.96783],
[114.27957,31.96795],
[114.27954,31.96825],
[114.27937,31.96778],
])


km_data = ma.asarray(ship_data)

km_data[19]=ma.masked

data=[
[114.27984,31.96803],
[114.27916,31.96783],
[114.28038,31.96796],
[114.28,31.96826],
[114.27974,31.96804],
[114.27954,31.96772],
[114.27964,31.96805],
[114.27974,31.9684	],
[114.28002,31.96783],
[114.27958,31.96807],
[114.27936,31.968],
[114.27962,31.96773],
[114.2798,31.968],
[114.2799,31.96796],
[114.27968,31.96799],
[114.27948,31.96771],
[114.2796,31.96783],
[114.27957,31.96795],
[114.27954,31.96825],
[0.,0.]
]

mask=[
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[False, False],
[True,True]
]
# fill_value=1e+20
init_mean = np.mean(ship_data[0:18],axis=0)

kf = KalmanFilter(initial_state_mean=init_mean, n_dim_obs=2)

result,_ = kf.em(km_data).smooth(km_data)
print(result)
plt.scatter(ship_data[:,0],ship_data[:,1],label='true')

plt.scatter(result[:,0],result[:,1],label='klm')

plt.legend()
plt.show()

# 10m左右的误差