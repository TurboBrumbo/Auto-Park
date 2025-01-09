from django.urls import path
from . import views


urlpatterns = [
    path('cars/', views.cars_home, name='cars_home'),
    path('cars/create', views.car_create, name='car_create'),
    path('cars/download', views.CarsDownloadView.as_view(), name='cars_download'),
    path('cars/<str:pk>/update', views.CarUpdateView.as_view(), name='car_update'),
    path('cars/<str:pk>/delete', views.CarDeleteView.as_view(), name='car_delete'),
    path('cars/<str:pk>', views.CarDetailView.as_view(), name='car_details'),
    path('drivers/', views.drivers_home, name='drivers_home'),
    path('drivers/create', views.driver_create, name='driver_create'),
    path('drivers/download', views.DriversDownloadView.as_view(), name='drivers_download'),
    path('drivers/<int:pk>/update', views.DriverUpdateView.as_view(), name='driver_update'),
    path('drivers/<int:pk>/delete', views.DriverDeleteView.as_view(), name='driver_delete'),
    path('drivers/<int:pk>', views.DriverDetailView.as_view(), name='driver_details'),
    path('trips/', views.trips_home, name='trips_home'),
    path('trips/create', views.trip_create, name='trip_create'),
    path('trips/download', views.TripsDownloadView.as_view(), name='trips_download'),
    path('trips/<int:pk>/update', views.TripUpdateView.as_view(), name='trip_update'),
    path('trips/<int:pk>/delete', views.TripDeleteView.as_view(), name='trip_delete'),
    path('trips/<int:pk>', views.TripDetailView.as_view(), name='trip_details'),
]