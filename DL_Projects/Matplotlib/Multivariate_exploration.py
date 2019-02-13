import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

fuel_econ = pd.read_csv("/home/wasi/ML_FOLDER/Matplotlib/data/fuel_econ.csv")

sedan_class = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_class)

fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)

fuel_econ['trans_type'] = fuel_econ['trans'].apply(lambda x : x.split()[0])

print(fuel_econ.head())

np.random.seed(2018)

sample = np.random.choice(fuel_econ.shape[0], 200, replace=False)
fuel_econ_subset = fuel_econ.loc[sample]

ttype_markers = [['Automatic', 'o'],
                 ['Manual', '^']]

for ttype, marker in ttype_markers:
    plot_data = fuel_econ_subset.loc[fuel_econ_subset['trans_type']==ttype]
    sns.regplot(data=plot_data, x='displ', y="comb", x_jitter=0.04, fit_reg=False, marker = marker)


sns.regplot(data=fuel_econ_subset, x='displ', y="comb", x_jitter=0.04, fit_reg=False,
            scatter_kws={'s':fuel_econ_subset['co2'] / 2})
sizes = [200, 350, 500]
base_color = sns.color_palette()[0]
legend_obj = []
for s in sizes:
    legend_obj.append(plt.scatter([], [], s=s/2, color=base_color))
plt.xlabel('Displacement(1)')
plt.ylabel('Combined fuel Eff. (mpg)')
plt.legend(legend_obj, sizes, title='CO2 (g/mi)')
plt.show()


g = sns.FacetGrid(data=fuel_econ_subset, hue="trans_type", hue_order=['Automatic', 'Manual'], size=4, aspect=1.5)
g.map(sns.regplot, 'displ', 'comb', x_jitter = 0.04, fit_reg = False)
plt.legend()
plt.xlabel("Displacement (l)")
plt.ylabel("Combined fuel efficiency")
plt.show()

# Note don't use numeric variables  bcoz facet expects only categorical variables other wise you will get messy legend..
g = sns.FacetGrid(data=fuel_econ_subset, hue="VClass", size=4, aspect=1.5, palette='viridis_r')
g.map(sns.regplot, 'displ', 'comb', x_jitter = 0.04, fit_reg = False)
plt.legend()
plt.xlabel("Displacement (l)")
plt.ylabel("Combined fuel efficiency")
plt.show()

# insted use scatter plot when we have more than 2 variables

plt.scatter(data=fuel_econ_subset, x='displ', y='comb', c='co2', cmap='viridis_r')
plt.colorbar(label="Co2 (mpg)")
plt.xlabel('Displacement')
plt.ylabel("Combined fuel efficiency")
plt.show()


# Using Facetgrid with two variables to plot.
g = sns.FacetGrid(data=fuel_econ_subset, col='VClass', row='trans_type', margin_titles=True)
g.map(plt.scatter, 'displ', 'comb')
plt.show()


# Other Adaptations of Bivariate Plots
# point plot
sns.pointplot(data=fuel_econ, x = 'VClass', y = "comb", hue="trans_type", ci="sd", linestyles="", dodge=True)
plt.xlabel("Displacement")
plt.ylabel("Combined fuel efficiency")
plt.xticks(rotation=20)
plt.show()

# barplot
sns.barplot(data=fuel_econ, x='VClass', y='comb', hue='trans_type', ci='sd')
plt.xlabel("Displacement")
plt.ylabel("Combined fuel efficiency")
plt.xticks(rotation =20)
plt.show()

# boxplot
sns.boxplot(data=fuel_econ, x='VClass', y='comb', hue='trans_type')
plt.xlabel("Displacement")
plt.ylabel("Combined fuel efficiency")
plt.xticks(rotation =20)
plt.show()

pokemon = pd.read_csv('/home/wasi/ML_FOLDER/Matplotlib/data/pokemon.csv')
print(pokemon.shape)
print(pokemon.head())
