from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd

from recommendations.logic import (
    get_recommendations,
    generate_visualizations,
    get_dataset,
    perform_apriori_analysis,
)
from recommendations.models import Recommendation


def apriori_results(request):
    (
        frequent_itemsets,
        filtered_itemsets,
        filtered_length3,
        sorted_rules,
    ) = perform_apriori_analysis()

    return render(
        request,
        "recommendations/apriori_results.html",
        {
            "frequent_itemsets": frequent_itemsets.to_html(),
            "filtered_itemsets": filtered_itemsets.to_html(),
            "filtered_length3": filtered_length3.to_html(),
            "sorted_rules": sorted_rules.to_html(),
        },
    )


def recommend(request):
    all_items = get_all_items()

    if request.method == "POST":
        input_data = request.POST.getlist("items[]")
        recommendations = get_recommendations(input_data)
        Recommendation.objects.create(items=input_data, recommendations=recommendations)
        return JsonResponse(
            {"recommendations": recommendations, "all_items": all_items}
        )
    return render(request, "recommendations/recommend.html", {"all_items": all_items})


def get_all_items():
    data = pd.read_csv(
        "https://res.cloudinary.com/devowino/raw/upload/v1700430513/Data%20Mining/Market_Basket_Optimisation_yqk5ci.csv"
    )
    all_items = data.stack().unique().tolist()
    return all_items


def render_visualizations(request):
    dataset = get_dataset()
    bar_chart_html, tree_map_html = generate_visualizations(dataset)

    return render(
        request,
        "recommendations/visualizations.html",
        {
            "bar_chart_html": bar_chart_html,
            "tree_map_html": tree_map_html,
        },
    )
