from mlxtend.frequent_patterns import association_rules, apriori
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import numpy as np


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

    frequent_itemsets = apriori(dataset, min_support=0.01, use_colnames=True)
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
