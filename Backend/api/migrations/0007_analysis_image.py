# Generated by Django 5.1.4 on 2024-12-29 18:33

import api.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0006_alter_patient_id"),
    ]

    operations = [
        migrations.AddField(
            model_name="analysis",
            name="image",
            field=models.ImageField(
                default=api.models.default_image_path, upload_to="analysis_images/"
            ),
        ),
    ]
