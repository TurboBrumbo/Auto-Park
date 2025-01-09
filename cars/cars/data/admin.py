from django.contrib.admin import site
from .models import Car, Driver, Trip
from import_export.resources import ModelResource


class CarResource(ModelResource):
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


class DriverResource(ModelResource):
    class Meta:
        model = Driver
        fields = [
            'id',
            'initials',
            'age',
            'experience'
        ]


class TripResource(ModelResource):
    class Meta:
        model = Trip
        fields = [
            'car_id',
            'driver_id',
            'length',
            'date',
            'rating'
        ]


site.register(Car)
site.register(Driver)
site.register(Trip)
