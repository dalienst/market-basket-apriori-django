from django.db import models


class Recommendation(models.Model):
    items = models.JSONField()
    recommendations = models.JSONField()
