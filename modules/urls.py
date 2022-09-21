from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    #path('avatar3d/', views.avatar3d, name='avatar3d'),
]