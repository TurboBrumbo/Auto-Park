from django.core.management import BaseCommand
from data.models import Car, Driver, Trip
import pandas as pd


class Command(BaseCommand):
    def handle(self, *args, **options):
        for model, link in zip([Car, Driver, Trip], ['./cars.csv', './drivers.csv', './trips.csv']):
            print(f'Loading data from {link}...')
            data = pd.read_csv(link)
            columns = list(data.columns)
            if link != './trips.csv':
                model.objects.bulk_create([
                    model(**dict(zip(columns, row.values))) for _, row in data.iterrows()
                ])
            else:
                for delete_column in 'car_id', 'driver_id':
                    columns.remove(delete_column)
                model.objects.bulk_create([
                    model(**{**{
                        'car_id': Car.objects.get(pk=row['car_id']),
                        'driver_id': Driver.objects.get(pk=row['driver_id'])
                    }, **dict(zip(columns, row.drop(['car_id', 'driver_id']).values))}) \
                        for _, row in data.iterrows()
                ])
