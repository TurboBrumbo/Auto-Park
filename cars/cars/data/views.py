from django.http import HttpResponse
from django.shortcuts import render, redirect
from .models import Car, Driver, Trip
from django.views.generic import DetailView, UpdateView, DeleteView, ListView, FormView
from .forms import CarForm, DriverForm, TripForm, DatasetDownloadForm
from .admin import CarResource, DriverResource, TripResource
import pandas as pd
import joblib


model = joblib.load('./model.joblib')


def cars_home(request):
    cars = Car.objects.all().order_by('id')
    return render(request, 'data/cars_home.html', {'cars': cars})


class CarDetailView(DetailView):
    model = Car
    template_name = 'data/car_details.html'
    context_object_name = 'car'
    needed_fields = [field.name for field in model._meta.get_fields() if field.name not in ['trip', 'id']]

    def get_context_data(self, **kwargs):
        context = super(CarDetailView, self).get_context_data(**kwargs)
        context['extra_qs'] = int(model.predict(pd.DataFrame({
            field: [self.get_object().__getattribute__(field)] for field in self.needed_fields
        }))[0])
        return context


class CarUpdateView(UpdateView):
    model = Car
    template_name = 'data/car_update.html'
    form_class = CarForm


class CarDeleteView(DeleteView):
    model = Car
    template_name = 'data/car_delete.html'
    success_url = '/data/cars/'


class CarsDownloadView(ListView, FormView):
    model = Car
    template_name = 'data/download.html'
    form_class = DatasetDownloadForm

    def post(self, request, **kwargs):
        queryset = self.get_queryset()
        dataset = CarResource().export(queryset)
        format = request.POST.get('format')

        if format == 'xlsx': dataset = dataset.xlsx
        elif format == 'csv': dataset = dataset.csv
        elif format == 'json': dataset = dataset.json

        response = HttpResponse(dataset, content_type=f'{format}')
        response['Content-Disposition'] = f'attachment; filename=cars.{format}'
        return response


def car_create(request):
    error = ''
    if request.method == 'POST':
        form = CarForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('cars_home')
        else:
            error = 'Форма была заполнена некорректно'
    form = CarForm()
    data = {
        'form': form,
        'error': error
    }
    return render(request, 'data/car_create.html', data)


def drivers_home(request):
    drivers = Driver.objects.all().order_by('id')
    return render(request, 'data/drivers_home.html', {'drivers': drivers})


class DriverDetailView(DetailView):
    model = Driver
    template_name = 'data/driver_details.html'
    context_object_name = 'driver'


class DriverUpdateView(UpdateView):
    model = Driver
    template_name = 'data/driver_update.html'
    form_class = DriverForm


class DriverDeleteView(DeleteView):
    model = Driver
    template_name = 'data/driver_delete.html'
    success_url = '/data/drivers/'


class DriversDownloadView(ListView, FormView):
    model = Driver
    template_name = 'data/download.html'
    form_class = DatasetDownloadForm

    def post(self, request, **kwargs):
        queryset = self.get_queryset()
        dataset = DriverResource().export(queryset)
        format = request.POST.get('format')

        if format == 'xlsx': dataset = dataset.xlsx
        elif format == 'csv': dataset = dataset.csv
        elif format == 'json': dataset = dataset.json

        response = HttpResponse(dataset, content_type=f'{format}')
        response['Content-Disposition'] = f'attachment; filename=drivers.{format}'
        return response


def driver_create(request):
    error = ''
    if request.method == 'POST':
        form = DriverForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('drivers_home')
        else:
            error = 'Форма была заполнена некорректно'
    form = DriverForm()
    data = {
        'form': form,
        'error': error
    }
    return render(request, 'data/driver_create.html', data)


def trips_home(request):
    trips = Trip.objects.all().order_by('-date')
    return render(request, 'data/trips_home.html', {'trips': trips})


class TripDetailView(DetailView):
    model = Trip
    template_name = 'data/trip_details.html'
    context_object_name = 'trip'


class TripUpdateView(UpdateView):
        model = Trip
        template_name = 'data/trip_update.html'
        form_class = TripForm


class TripDeleteView(DeleteView):
        model = Trip
        template_name = 'data/trip_delete.html'
        success_url = '/data/trips/'


class TripsDownloadView(ListView, FormView):
    model = Trip
    template_name = 'data/download.html'
    form_class = DatasetDownloadForm

    def post(self, request, **kwargs):
        queryset = self.get_queryset()
        dataset = TripResource().export(queryset)
        format = request.POST.get('format')

        if format == 'xlsx': dataset = dataset.xlsx
        elif format == 'csv': dataset = dataset.csv
        elif format == 'json': dataset = dataset.json

        response = HttpResponse(dataset, content_type=f'{format}')
        response['Content-Disposition'] = f'attachment; filename=trips.{format}'
        return response


def trip_create(request):
    error = ''
    if request.method == 'POST':
        form = TripForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('trips_home')
        else:
            error = 'Форма была заполнена некорректно'
    form = TripForm()
    data = {
        'form': form,
        'error': error
    }
    return render(request, 'data/trip_create.html', data)
