from typing import Any
from django import forms
from django.core.exceptions import ValidationError

class ImageForm(forms.Form):
    images = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={"class": "form-check-input"}))
    group = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={"class": "form-check-input"}))
    description = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={"class": "form-check-input"}))
    image = forms.ImageField(required=True, widget=forms.FileInput(attrs={"class": "form-control form-control-lg"}),
                             error_messages={"invalid_image": "Файл не является изображением"})

    def clean(self) -> dict[str, Any]:
        cd = self.cleaned_data

        images = cd.get("images")
        group = cd.get("group")
        description = cd.get("description")

        if images + group + description < 1:
            #Or you might want to tie this validation to the password1 field
            raise ValidationError("Хотя-бы одна опция должна быть выбрана")
        return cd
