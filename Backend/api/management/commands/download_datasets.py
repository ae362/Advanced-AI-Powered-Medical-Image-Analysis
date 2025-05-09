from django.core.management.base import BaseCommand
import os
import shutil
from pathlib import Path
import subprocess
import zipfile

class Command(BaseCommand):
    help = 'Download and organize datasets from Kaggle'

    def handle(self, *args, **options):
        try:
            # Ensure Kaggle CLI is in PATH
            kaggle_command = shutil.which("kaggle")
            if kaggle_command is None:
                raise FileNotFoundError("Kaggle CLI not found. Please ensure it is installed and in the system PATH.")
            
            self.stdout.write(f"Kaggle command found at: {kaggle_command}")

            # Create base directories
            self.stdout.write('Creating directories...')
            base_dir = Path('data')
            directories = [
                base_dir / 'brain_tumor' / 'positive',
                base_dir / 'brain_tumor' / 'negative',
                base_dir / 'brain_tumor' / 'test',
                base_dir / 'cancer' / 'positive',
                base_dir / 'cancer' / 'negative',
                base_dir / 'cancer' / 'test',
                base_dir / 'temp' / 'brain_tumor',
                base_dir / 'temp' / 'cancer'
            ]

            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.stdout.write(f'Created directory: {directory}')

            # Download datasets using Kaggle CLI
            self.stdout.write('Downloading brain tumor dataset...')
            subprocess.run([
                kaggle_command, 'datasets', 'download', '-d', 'alaminbhuyan/mri-image-data',
                '-p', str(base_dir / 'temp' / 'brain_tumor')
            ], check=True)

            self.stdout.write('Downloading lung cancer dataset...')
            subprocess.run([
                kaggle_command, 'datasets', 'download', '-d', 'hamdallak/the-iqothnccd-lung-cancer-dataset',
                '-p', str(base_dir / 'temp' / 'cancer')
            ], check=True)

            # Extract datasets
            self.extract_and_organize_datasets(base_dir)

            # Clean up temp directory
            shutil.rmtree(base_dir / 'temp')
            self.stdout.write(self.style.SUCCESS('Datasets downloaded, extracted, and organized successfully'))

        except subprocess.CalledProcessError as e:
            self.stdout.write(self.style.ERROR(f'Kaggle CLI Error: {e}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Unexpected error: {str(e)}'))

    def extract_and_organize_datasets(self, base_dir):
        # Extract and organize brain tumor dataset
        brain_tumor_zip = next(Path(base_dir / 'temp' / 'brain_tumor').glob('*.zip'))
        with zipfile.ZipFile(brain_tumor_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir / 'temp' / 'brain_tumor')
        
        # Extract and organize cancer dataset
        cancer_zip = next(Path(base_dir / 'temp' / 'cancer').glob('*.zip'))
        with zipfile.ZipFile(cancer_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir / 'temp' / 'cancer')

        # Process brain tumor dataset
        brain_dataset_path = base_dir / 'temp' / 'brain_tumor' / 'Brain Tumor'
        if not brain_dataset_path.exists():
            brain_dataset_path = base_dir / 'temp' / 'brain_tumor'  # Try alternative path

        # Print directory contents for debugging
        self.stdout.write("Brain tumor directory contents:")
        self.print_directory_contents(brain_dataset_path)

        # Process Training directory
        training_path = brain_dataset_path / 'Training'
        if training_path.exists():
            # Process no tumor images
            no_tumor_path = training_path / 'notumor'
            if no_tumor_path.exists():
                for img in no_tumor_path.glob('*.*'):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy2(img, base_dir / 'brain_tumor' / 'negative' / img.name)
                        self.stdout.write(f'Copied negative image: {img.name}')

            # Process tumor images
            for tumor_type in ['glioma', 'meningioma', 'pituitary']:
                tumor_path = training_path / tumor_type
                if tumor_path.exists():
                    for img in tumor_path.glob('*.*'):
                        if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            shutil.copy2(img, base_dir / 'brain_tumor' / 'positive' / img.name)
                            self.stdout.write(f'Copied positive image: {img.name}')

        # Process cancer dataset
        cancer_dataset_path = base_dir / 'temp' / 'cancer'
        
        # Print directory contents for debugging
        self.stdout.write("Cancer directory contents:")
        self.print_directory_contents(cancer_dataset_path)

        # Process normal cases
        normal_path = cancer_dataset_path / 'Normal cases'
        if normal_path.exists():
            for img in normal_path.glob('*.*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img, base_dir / 'cancer' / 'negative' / img.name)
                    self.stdout.write(f'Copied negative image: {img.name}')

        # Process benign and malignant cases
        for case_type in ['Bengin cases', 'Malignant cases']:
            case_path = cancer_dataset_path / case_type
            if case_path.exists():
                for img in case_path.glob('*.*'):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy2(img, base_dir / 'cancer' / 'positive' / img.name)
                        self.stdout.write(f'Copied positive image: {img.name}')

        # Create test sets
        self.create_test_set(base_dir, 'brain_tumor')
        self.create_test_set(base_dir, 'cancer')

    def print_directory_contents(self, path):
        """Helper function to print directory contents for debugging"""
        try:
            self.stdout.write(f"\nContents of {path}:")
            for item in path.iterdir():
                self.stdout.write(f"- {item.name}")
                if item.is_dir():
                    for subitem in item.iterdir():
                        self.stdout.write(f"  - {subitem.name}")
        except Exception as e:
            self.stdout.write(f"Error reading directory {path}: {str(e)}")

    def create_test_set(self, base_dir, dataset_type):
        """Move 20% of images to test set"""
        import random
        
        pos_dir = base_dir / dataset_type / 'positive'
        neg_dir = base_dir / dataset_type / 'negative'
        test_dir = base_dir / dataset_type / 'test'
        
        for source_dir in [pos_dir, neg_dir]:
            if source_dir.exists():
                images = list(source_dir.glob('*.[pj][np][g]'))
                if images:
                    test_size = max(1, int(len(images) * 0.2))
                    test_images = random.sample(images, test_size)
                    
                    for img_path in test_images:
                        dest = test_dir / img_path.name
                        shutil.move(str(img_path), str(dest))
                        self.stdout.write(f'Moved to test set: {img_path.name}')