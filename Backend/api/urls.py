from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import PatientViewSet, AnalysisViewSet,TrainingViewSet,register_user,CustomTokenObtainPairView,logout_view,send_analysis_email,send_analysis_to_email
from . import views
from rest_framework_simplejwt.views import TokenRefreshView
router = DefaultRouter()
router.register(r'patients', PatientViewSet)
router.register(r'analyses', AnalysisViewSet)
router.register(r'training', TrainingViewSet, basename='training')
urlpatterns = [
    path('', include(router.urls)),
    path('register/', register_user, name='register'),
    path('diseases/', views.disease_list, name='disease-list'),
    path('diseases/<int:disease_id>/upload-model/', views.upload_model, name='upload-model'),
    path('analyses/<uuid:pk>/notes/', views.update_analysis_notes, name='update-analysis-notes'),
    path('token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('logout/', logout_view, name='auth_logout'),
    path('analyses/<str:analysis_id>/explanation-and-staging/', views.generate_explanation_and_staging, name='generate_explanation'),
    path('analyses/<str:analysis_id>/update-email/', views.update_notification_email, name='update_notification_email'),    
    path('analyses/<str:analysis_id>/send-email/', send_analysis_email, name='send_analysis_email'),
    path('analyses/<str:analysis_id>/send-to-email/',  send_analysis_to_email,  name='send_analysis_to_email'),
]