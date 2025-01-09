from django.db.models import (
    Model,
    IntegerField,
    PositiveIntegerField,
    DecimalField,
    CharField,
    TextField,
    DateField,
    ForeignKey,
    CASCADE
)


class Car(Model):
    id = CharField(max_length=6, primary_key=True)
    mileage = IntegerField()
    vehicle_age = IntegerField()
    reported_issues = IntegerField()
    fuel_efficiency = DecimalField(decimal_places=15, max_digits=17)
    service_history = IntegerField()
    accident_history = IntegerField()
    tire_condition = IntegerField()
    brake_condition = IntegerField()
    battery_status = IntegerField()
    days_since_last_service = IntegerField()
    days_to_warranty_expire = IntegerField()

    def __str__(self):
        return f'{self.id}'

    def get_absolute_url(self):
        return f'/data/cars/{self.id}'

    class Meta:
        verbose_name = 'Автомобиль'
        verbose_name_plural = 'Автомобили'


class Driver(Model):
    id = IntegerField(primary_key=True)
    initials = TextField()
    age = IntegerField()
    experience = IntegerField()

    def __str__(self):
        return f'{self.id}'

    def get_absolute_url(self):
        return f'/data/drivers/{self.id}'

    class Meta:
        verbose_name = 'Водитель'
        verbose_name_plural = 'Водители'


class Trip(Model):
    car_id = ForeignKey(Car, on_delete=CASCADE, to_field='id')
    driver_id = ForeignKey(Driver, on_delete=CASCADE, to_field='id')
    length = DecimalField(decimal_places=1, max_digits=5)
    date = DateField()
    rating = PositiveIntegerField()

    def __str__(self):
        return f'{self.id}'

    def get_absolute_url(self):
        return f'/data/trips/{self.id}'

    class Meta:
        verbose_name = 'Поездка'
        verbose_name_plural = 'Поездки'
