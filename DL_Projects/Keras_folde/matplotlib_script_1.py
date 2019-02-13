import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pokemon = pd.read_csv("/home/wasi/ML_FOLDER/pokemon.csv")
print(pokemon.shape)
print(pokemon.head())


base_color = sns.color_palette()[4]
gen_order = pokemon['generation_id'].value_counts().index

sns.countplot(data=pokemon, x='generation_id', color=base_color, order=gen_order)
plt.show()

type_order = pokemon['type_1'].value_counts().index

sns.countplot(data=pokemon, y='type_1', color=base_color, order=type_order)
plt.show()

pkmn_types = pokemon.melt(id_vars=['id', 'species'],
                          value_vars = ['type_1', 'type_2'],
                          var_name = 'type_level', value_name = 'type').dropna()

print(pkmn_types[802:812])

type_counts = pkmn_types['type'].value_counts()
print(type_counts)
type_order = type_counts.index

sns.countplot(data=pkmn_types, y='type', color=base_color, order=type_order)
plt.show()


n_pokemon  = pokemon.shape[0].astype(np.int64)
print(n_pokemon)
max_type_count = type_counts[0]
max_type_count_changed = max_type_count.astype(np.int)
print(max_type_count)
max_prop = max_type_count_changed / n_pokemon
print(max_prop)
