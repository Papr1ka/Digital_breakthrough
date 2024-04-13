from django import forms

class ImageForm(forms.Form):
    image = forms.ImageField(required=True, widget=forms.FileInput(attrs={"class": "form-control form-control-lg"}))
