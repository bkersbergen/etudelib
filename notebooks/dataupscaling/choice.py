#%% Import packages
import numpy as np
import pandas as pd
from pathlib import Path
#%% Read data
current_path = Path(__file__).parent
df = pd.read_csv(f"{current_path}/freqs.csv")
#%% Compute frequencies and probabilities
frequencies = df.groupby(["qty"]).count()
frequencies.columns = ["frequency"]
probabilities = frequencies / np.sum(frequencies)
#%% Simple upscaler based on empirical probabilities
seed = 0
n_random_items = 50_000_000
rng = np.random.default_rng(seed=seed)
random_items = rng.choice(probabilities.index.values, 
                          size=n_random_items,
                          replace=True,
                          p = probabilities.values.squeeze())

# The vector random_items contains the frequency of each item
# and the item-ID is simply the index in this vector, i.e
# 'item with id 0 occurs random_items[0] = x times'
# 'item with id 1 occurs random_items[1] = y times'
# 'item with id 2 occurs random_items[2] = z times'
# etc
#%% Do idiot check that the simulated probability equals the observed probability
# frequency = df["qty"].max()
# frequency = df["qty"].min()
frequency = 100
# How many items with this frequency occur in our new vector?
occurence = len(random_items[random_items == frequency])
simulated_probability = occurence / n_random_items
# Check probability (should be approx. equal)
print(f"Empirical probability: {probabilities.loc[frequency, 'frequency']:.7f}")
print(f"Simulated probability: {simulated_probability:.7f}")
