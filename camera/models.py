from django.db import models

# Create your models here.


class Image(models.Model):
    title = models.TextField(max_length=20)
    file = models.FileField(upload_to='img')
    
    def __str__(self) -> str:
        return self.title