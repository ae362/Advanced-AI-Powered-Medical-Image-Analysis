from django.db import models
import uuid
import random
from django.db.models.signals import pre_save
from django.contrib.auth.models import AbstractUser
from django.dispatch import receiver
from .utilities.encryption import encrypt_data, decrypt_data
def generate_patient_id():
    while True:
        new_id = str(random.randint(1000000, 9999999))
        if not Patient.objects.filter(id=new_id).exists():
            return new_id
class Patient(models.Model):
    id = models.CharField(max_length=7, primary_key=True, default=generate_patient_id, editable=False)
    name = models.CharField(max_length=255)
    date_of_birth = models.DateField()
    gender = models.CharField(max_length=10, choices=[
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other')
    ])
    contact_details = models.CharField(max_length=255)
    email = models.EmailField(blank=True, null=True)
    medical_history = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"{self.name} (ID: {self.id})"
    
    def set_medical_history(self, medical_history):
        self.medical_history = encrypt_data(medical_history)
    
    def get_medical_history(self):
        return decrypt_data(self.medical_history)


def default_image_path():
    return 'analysis_images/default.png'
class Analysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='analyses')
    notes = models.TextField(blank=True, null=True)
    type = models.CharField(max_length=20, choices=[
        ('brain_tumor', 'Brain Tumor'),
        ('cancer', 'Cancer')
    ])
    prediction = models.CharField(max_length=20)
    confidence = models.FloatField()
    image = models.ImageField(upload_to='analysis_images/', default=default_image_path)
    visualization = models.TextField()  # Base64 encoded image
    notification_email = models.EmailField(null=True, blank=True)
    
    model_accuracy = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.patient.name} - {self.type} Analysis"

class Disease(models.Model):
    name = models.CharField(max_length=255)
    key = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    model_file = models.FileField(upload_to='disease_models/', null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)  # Only active when model is uploaded

    def __str__(self):   
        return self.name


class TrainingClass(models.Model):
    disease = models.ForeignKey(Disease, on_delete=models.CASCADE, related_name='classes')
    key = models.CharField(max_length=50)  # e.g., 'positive', 'negative'
    name = models.CharField(max_length=255)  # e.g., 'Positive Cases', 'Negative Cases'
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['disease', 'key']

    def __str__(self):
        return f"{self.disease.name} - {self.name}"

class TrainingImage(models.Model):
    training_class = models.ForeignKey(
        TrainingClass, 
        on_delete=models.CASCADE, 
        related_name='images',
        null=True,  # Allow null temporarily for migration
        blank=True  # Allow blank in forms
    )
    image = models.ImageField(upload_to='training_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Training image for {self.training_class or 'unassigned'}"

class CustomUser(AbstractUser):
    is_admin = models.BooleanField(default=False)

    # Add related_name arguments to resolve clashes
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name='groups',
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        related_name='customuser_set',
        related_query_name='customuser',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name='customuser_set',
        related_query_name='customuser',
    )

    def __str__(self):
        return self.username
