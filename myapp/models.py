from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class N(models.Model):
    category_name = models.CharField(max_length=255)


class Userprofile(models.Model):
    USER=models.OneToOneField(User,on_delete=models.CASCADE)
    photo = models.FileField(max_length=255,upload_to='userphotos/')
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    gender = models.CharField(max_length=25)
    email = models.CharField(max_length=255)
    address = models.CharField(max_length=255)
    place = models.CharField(max_length=50)
    phone = models.CharField(max_length=20)
    status = models.CharField(max_length=20,default="active")

class phishing_reports(models.Model):
    USERPROFILE = models.ForeignKey(Userprofile, on_delete=models.CASCADE)
    url = models.CharField(max_length=500)
    prediction = models.CharField(max_length=50)
    explanation = models.TextField(blank=True, null=True)
    scan_date = models.CharField(max_length=50)

class spam_email_reports(models.Model):
    USERPROFILE = models.ForeignKey(Userprofile, on_delete=models.CASCADE)
    email_text = models.TextField()
    prediction = models.CharField(max_length=50)
    scan_date = models.CharField(max_length=50)


class image_analysis(models.Model):
    USERPROFILE = models.ForeignKey(Userprofile, on_delete=models.CASCADE)
    image_path = models.FileField(max_length=255)
    prediction = models.TextField()
    model_results = models.TextField(blank=True, null=True)  # JSON: per-model breakdown
    upload_date = models.CharField(max_length=50)

class complaints(models.Model):
    USERPROFILE = models.ForeignKey(Userprofile, on_delete=models.CASCADE)
    complaint_text = models.CharField(max_length=1000)
    reply_text = models.CharField(max_length=1000, blank=True, null=True)
    status = models.CharField(max_length=20)
    submitted_at = models.CharField(max_length=50)

class notifications(models.Model):
    title = models.CharField(max_length=255)
    message = models.CharField(max_length=1000)
    created_at = models.CharField(max_length=50)

class NotificationRead(models.Model):
    notification = models.ForeignKey(notifications, on_delete=models.CASCADE)
    user = models.ForeignKey(Userprofile, on_delete=models.CASCADE)
    is_read = models.BooleanField(default=False)