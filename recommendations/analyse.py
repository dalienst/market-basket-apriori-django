# or convert to .ipynb

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

# Loading the data
data = pd.read_csv(
    "/content/drive/MyDrive/MACHINE LEARNING/Market_Basket_Optimisation.csv"
)

# Gathering all items into a list of lists
transactions = []
for i in range(data.shape[0]):
    transaction = [item for item in data.iloc[i].dropna()]
    transactions.append(transaction)

# Initializing the TransactionEncoder
te = TransactionEncoder()

# Using the TransactionEncoder to fit and transform the transactions
te_ary = te.fit(transactions).transform(transactions)

# Creating a DataFrame with the encoded transactions
dataset = pd.DataFrame(te_ary, columns=te.columns_)

# Applying Apriori algorithm
frequent_itemsets = apriori(dataset, min_support=0.01, use_colnames=True)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

# Print the most frequent itemsets
print("Performing Apriori: Extracting the most frequent itemsets:")
print(frequent_itemsets)

# Filtering the itemsets based on length and support
filtered_itemsets = frequent_itemsets[
    (frequent_itemsets["length"] == 2) & (frequent_itemsets["support"] >= 0.05)
]

filtered_length3 = frequent_itemsets[(frequent_itemsets["length"] == 3)].head(3)

# Print the filtered itemsets
print("\nItems with a length of 2 and minimum support of more than 0.05:")
print(filtered_itemsets)

print("\nItems with a length of 3:")
print(filtered_length3)

"""
Mining association rules
"""
# Creating antecedents and consequences
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))

# Sort values based on confidence
sorted_rules = rules.sort_values("confidence", ascending=False)

# Print the sorted rules
print("\nAssociation Rules sorted by confidence:")
print(sorted_rules)

"""
Data Visualization
"""

# Create a DataFrame for visualization
df_table = (
    dataset.sum(axis=0)
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"index": "items", 0: "item_count"})
)

# Initial Visualizations
print("\nTop 50 items visualization:")
df_table.head(50).style.background_gradient(cmap="Greens")
print(" ")

# Tree mapping the dataset to visualize it more interactively
df_table["all"] = "all"
fig = px.treemap(
    df_table.head(30),
    path=["all", "items"],
    values="item_count",
    color=df_table["item_count"].head(30),
    hover_data=["items"],
    color_continuous_scale="Reds",
)
fig.show()
