from django.urls import path
from . import views

urlpatterns = [
    path('home', views.home, name='home'),
     path('', views.home, name='home'),
    path('register', views.register, name='register'),
    path('getfile', views.getfile, name='getfile'),
   
    path('make_plot', views.PlotView.as_view(), name='make_plot'),
]




