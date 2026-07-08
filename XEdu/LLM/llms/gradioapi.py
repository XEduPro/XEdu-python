import os
from .base import BaseLLM
import warnings
try:
    from gradio_client import Client
except ImportError as exc:
    raise ImportError(
        "Using an xedu_url requires the optional dependency 'gradio-client'. "
        "Install it before creating a GradioClient."
    ) from exc
    
class GradioClient(BaseLLM):
    def __init__(self, base_url):
        self.base_url = base_url
        self.client = Client(base_url)


    def inference(self, message,api_name="/chat",**kwargs):
        res = self.client.predict(message,api_name=api_name)
        return res
    
    def list_models(self):
        warnings.warn("Due to using the local xedu URL, you do not need to specify a model and cannot view the model list.")
        return []
