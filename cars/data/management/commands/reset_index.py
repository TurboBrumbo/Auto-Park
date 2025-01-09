from django.core.management import BaseCommand
from django.db import connection
from data.models import Trip


class Command(BaseCommand):
    def handle(self, *args, **options):
        with connection.cursor() as cursor:
            cursor.execute(f"UPDATE sqlite_sequence SET seq = 0 WHERE name='{Trip._meta.db_table}'")
        print('Index of table Trip was reset successfully')
