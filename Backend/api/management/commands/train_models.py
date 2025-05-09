from django.core.management.base import BaseCommand
from api.ml.train import main as train_main

class Command(BaseCommand):
    help = 'Train the medical image analysis models'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting model training...'))
        train_main()
        self.stdout.write(self.style.SUCCESS('Model training completed successfully!'))

