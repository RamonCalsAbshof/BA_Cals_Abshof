from django.urls import path
from parameter_handler import views
from django.conf import settings
from django.conf.urls.static import static

from rest_framework.urlpatterns import format_suffix_patterns
from .dockerHandler import DockerHandler
from .initializeDBtables import InitializeDBtables
from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
        path('submit/', csrf_exempt(views.SubmitView.as_view())),
        path('algorithms/', csrf_exempt(views.AlgorithmView.as_view())),
        path('tablenames/', csrf_exempt(views.TablenameView.as_view())),
        path('tabledata/', csrf_exempt(views.TabledataView.as_view())),
        path('run/<int:run_id>/', csrf_exempt(views.RunView.as_view())),
        ]

urlpatterns = format_suffix_patterns(urlpatterns)
InitializeDBtables.compareAlgorithmsAndTables()



