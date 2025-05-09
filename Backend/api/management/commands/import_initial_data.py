import os
from django.core.management.base import BaseCommand
from django.core.files import File
from api.models import MedicalImage

class Command(BaseCommand):
    help = 'Import initial dataset from Kaggle'

    def add_arguments(self, parser):
        parser.add_argument('data_dir', type=str, help='Path to the data directory')

    def handle(self, *args, **options):
        data_dir = options['data_dir']
        
        for detection_type in ['brain_tumor', 'cancer']:
            for class_type in ['positive', 'negative']:
                class_dir = os.path.join(data_dir, detection_type, class_type)
                if not os.path.exists(class_dir):
                    self.stdout.write(self.style.WARNING(f"Directory not found: {class_dir}"))
                    continue

                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    with open(img_path, 'rb') as img_file:
                        medical_image = MedicalImage(
                            detection_type=detection_type,
                            ground_truth=class_type,
                            used_for_training=False
                        )
                        medical_image.image.save(img_name, File(img_file), save=True)
                    
                    self.stdout.write(self.style.SUCCESS(f"Imported: {img_path}"))

        self.stdout.write(self.style.SUCCESS("Initial data import completed"))