from django.core.management.base import BaseCommand
from api.models import Patient

class Command(BaseCommand):
    help = 'Updates patients with default IDs to have unique IDs'

    def handle(self, *args, **options):
        patients = Patient.objects.filter(patient_id='0000000')
        for patient in patients:
            patient.save()  # This will trigger the save method and generate a new ID
        self.stdout.write(self.style.SUCCESS(f'Successfully updated {patients.count()} patients'))

