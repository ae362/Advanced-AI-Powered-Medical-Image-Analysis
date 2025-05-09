from cryptography.fernet import Fernet
from django.conf import settings
import base64
import os

def get_or_create_key():
    key_file = os.path.join(settings.BASE_DIR, 'encryption_key.key')
    if os.path.exists(key_file):
        with open(key_file, 'rb') as file:
            key = file.read()
    else:
        key = Fernet.generate_key()
        with open(key_file, 'wb') as file:
            file.write(key)
    return key

ENCRYPTION_KEY = get_or_create_key()
fernet = Fernet(ENCRYPTION_KEY)

def encrypt_data(data):
    if isinstance(data, str):
        data = data.encode()
    return base64.urlsafe_b64encode(fernet.encrypt(data)).decode()

def decrypt_data(encrypted_data):
    if isinstance(encrypted_data, str):
        encrypted_data = encrypted_data.encode()
    return fernet.decrypt(base64.urlsafe_b64decode(encrypted_data)).decode()

