# Generated by Django 4.2 on 2025-01-12 18:06

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0014_analysis_stage_description"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="analysis",
            name="stage_description",
        ),
    ]
