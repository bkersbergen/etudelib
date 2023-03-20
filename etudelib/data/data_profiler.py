import scipy.stats as st
import numpy as np
import pandas as pd

columns = ['session_id', 'datetime_column', 'item_id', 'x']
usecols = ['session_id', 'item_id']
df = pd.read_csv('yoochoose-clicks.dat', header=None, names=columns, usecols=usecols)

distribution_name = 'powerlaw'
distribution = getattr(st, distribution_name)

# Observed frequencies
f_obs = df.groupby('session_id').size().to_numpy()
f_obs = np.sort(f_obs)

# Fit a distribution to a frequency
s_fit_params = distribution.fit(f_obs)
print('yoochoose-clicks.dat parameters for sessions')
print(s_fit_params)
# (0.10324840656752753, 0.9999999999999999, 199.00000000000003)

# Observed frequencies
f_obs = df.groupby('item_id').size().to_numpy()
f_obs = np.sort(f_obs)

# Fit a distribution to a frequency
i_fit_params = distribution.fit(f_obs)
print('yoochoose-clicks.dat parameters for items')
# (0.07980915552672938, 0.9999999999999999, 147418.00000000003)
