from .models import Car, Driver, Trip
from django.forms import (
    Form,
    ModelForm,
    ChoiceField,
    NumberInput,
    TextInput,
    DateInput,
    Select
)


class CarForm(ModelForm):
    class Meta:
        model = Car
        fields = [
            'id',
            'mileage',
            'vehicle_age',
            'reported_issues',
            'fuel_efficiency',
            'service_history',
            'accident_history',
            'tire_condition',
            'brake_condition',
            'battery_status',
            'days_since_last_service',
            'days_to_warranty_expire'
        ]

        widgets = {
            'id': TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Идентификатор автомобиля'
            }),
            'mileage': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Пробег',
            }),
            'vehicle_age': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Возраст',
            }),
            'reported_issues': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Количество поломок',
            }),
            'fuel_efficiency': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Мощность двигателя',
            }),
            'service_history': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Количество починок',
            }),
            'accident_history': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Количество аварий',
            }),
            'tire_condition': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Состояние шин',
            }),
            'brake_condition': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Состояние тормозов',
            }),
            'battery_status': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Состояние двигателя',
            }),
            'days_since_last_service': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Количество дней с последнего ТО',
            }),
            'days_to_warranty_expire': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Количество дней то истечения страховки',
            }),
        }


class DriverForm(ModelForm):
    class Meta:
        model = Driver
        fields = [
            'id',
            'initials',
            'age',
            'experience'
        ]

        widgets = {
            'id': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Идентификатор водителя'
            }),
            'initials': TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Инициалы',
            }),
            'age': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Возраст',
            }),
            'experience': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Опыт вождения',
            })
        }


class TripForm(ModelForm):
    class Meta:
        model = Trip
        fields = [
            'car_id',
            'driver_id',
            'length',
            'date',
            'rating'
        ]

        widgets = {
            'car_id': TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Идентификатор автомобиля'
            }),
            'driver_id': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Идентификатор водителя',
            }),
            'length': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Длина поездки',
            }),
            'date': DateInput(attrs={
                'class': 'form-control',
                'placeholder': 'Дата поездки',
            }),
            'rating': NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Оценка поездки',
            })
        }


choices = (
    ('txt', 'txt'),
    ('xls', 'xls'),
    ('csv', 'csv'),
    ('json', 'json')
)

class DatasetDownloadForm(Form):
    format = ChoiceField(choices=choices, widget=Select(attrs={'class': 'form-select'}))
