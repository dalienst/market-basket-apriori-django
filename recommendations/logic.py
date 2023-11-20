from mlxtend.frequent_patterns import association_rules, apriori
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import numpy as np


def perform_apriori_analysis(
    data_path="https://res.cloudinary.com/devowino/raw/upload/v1700430513/Data%20Mining/Market_Basket_Optimisation_yqk5ci.csv",
):
    """
    performing apriori analysis on a dataset
    """
    data = pd.read_csv(data_path)

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

    # Filtering the itemsets based on length and support
    filtered_itemsets = frequent_itemsets[
        (frequent_itemsets["length"] == 2) & (frequent_itemsets["support"] >= 0.05)
    ]

    # items with a length of 3
    filtered_length3 = frequent_itemsets[(frequent_itemsets["length"] == 3)].head(3)

    # Mining association rules
    # Creating antecedents and consequences
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))

    # Sort values based on confidence
    sorted_rules = rules.sort_values("confidence", ascending=False)

    return frequent_itemsets, filtered_itemsets, filtered_length3, sorted_rules


def get_recommendations(input_data):
    data = pd.read_csv(
        "https://res.cloudinary.com/devowino/raw/upload/v1700430513/Data%20Mining/Market_Basket_Optimisation_yqk5ci.csv"
    )
    # dropping null values
    transactions = [
        [item for item in data.iloc[i].dropna()] for i in range(data.shape[0])
    ]
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    dataset = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(dataset, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

    # For simplicity, just return the first 5 rules as recommendations
    return list(rules.head()["consequents"].apply(list))


def get_dataset():
    data = pd.read_csv(
        "https://res.cloudinary.com/devowino/raw/upload/v1700430513/Data%20Mining/Market_Basket_Optimisation_yqk5ci.csv"
    )
    # dropping null values
    transactions = [
        [item for item in data.iloc[i].dropna()] for i in range(data.shape[0])
    ]
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)


def generate_visualizations(dataset):
    # creating a dataframe
    df_table = (
        dataset.sum(axis=0)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "items", 0: "item_count"})
    )

    # create bar chart for top 10
    bar_chart = px.bar(
        df_table.head(10),
        x="items",
        y="item_count",
        title="Top 10 Items",
        labels={"item_count": "Item Count"},
    )

    # Create a tree map for the top 50 items
    df_table["all"] = "all"
    tree_map = px.treemap(
        df_table.head(50),
        path=["all", "items"],
        values="item_count",
        color=df_table["item_count"].head(50),
        hover_data=["items"],
        color_continuous_scale="Reds",
        title="Top 50 Items Tree Map",
    )

    # Render the visualizations as HTML
    bar_chart_html = bar_chart.to_html(full_html=False)
    tree_map_html = tree_map.to_html(full_html=False)

    return bar_chart_html, tree_map_html
