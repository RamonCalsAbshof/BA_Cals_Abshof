from django.urls import path
from parameter_handler import views

from rest_framework.urlpatterns import format_suffix_patterns
from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
        path('api/submit/', csrf_exempt(views.SubmitView.as_view())),
        path('api/algorithms/', csrf_exempt(views.AlgorithmView.as_view())),
        path('api/tablenames/', csrf_exempt(views.TablenameView.as_view())),
        path('api/tabledata/', csrf_exempt(views.TabledataView.as_view())),
        path('api/run/<int:run_id>/', csrf_exempt(views.RunView.as_view())),
        ]

urlpatterns = format_suffix_patterns(urlpatterns)

