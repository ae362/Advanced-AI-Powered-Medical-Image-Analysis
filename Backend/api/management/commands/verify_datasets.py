from django.core.management.base import BaseCommand
from pathlib import Path

class Command(BaseCommand):
    help = 'Verify the organization of datasets'

    def handle(self, *args, **options):
        datasets = ['brain_tumor', 'cancer']
        categories = ['positive', 'negative', 'test']

        for dataset in datasets:
            self.stdout.write(f"\nVerifying {dataset} dataset:")
            for category in categories:
                path = Path(f'data/{dataset}/{category}')
                if path.exists():
                    file_count = len(list(path.glob('*.[pj][np][g]')))
                    self.stdout.write(f"  {category}: {file_count} images")
                else:
                    self.stdout.write(self.style.WARNING(f"  {category} directory not found"))

        self.stdout.write(self.style.SUCCESS("\nDataset verification complete"))