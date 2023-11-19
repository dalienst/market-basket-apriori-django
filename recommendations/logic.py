from mlxtend.frequent_patterns import association_rules, apriori
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


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
