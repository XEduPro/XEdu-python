from .base import BaseLLM
import requests

class Qwen(BaseLLM):
    def __init__(self, api_key, model="qwen2-1.5b-instruct"):
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def _set_message(self, messages,stream=True):
        if stream:
            send_messages = {
                "model": self.model, # Optional
                "input":{
                    "messages": messages,
                },
                "parameters": {
                    "incremental_output": True,
                    "result_format": "message"
                }
            }
        else:
            send_messages = {
                "model": self.model, # Optional
                "input":{
                    "messages": messages,
                },
                "parameters": {
                    "result_format": "message"
                }
            }
        return send_messages
    
    def _set_content(self, response):
        return response.json()['output']["choices"][0]["message"]["content"]
    def _set_headers(self,stream=False):
        if stream:
            headers={
                'Content-Type': 'application/json',
                'X-DashScope-SSE': 'enable',
                "Authorization": f"Bearer {self.api_key}",
            }
        else:
            headers={
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {self.api_key}",
            }
        return headers
    
    def _analyse_stream_response(self, response):
        """
        解析推理结果
        """
        # pattern = re.compile(r'"content":"(.*?)","role"')

        # http_response = []
        import json
        for chunk in response.iter_lines(chunk_size=None):
            chunk = chunk.decode('utf-8')
            if chunk.startswith('data:'):
                ch_dict = json.loads(chunk[5:])
                yield ch_dict['output']['choices'][0]['message']['content']
    
    def _set_url(self):
        return self.base_url
    
    def list_models(self):
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        url = base_url + "/models"
        payload={}
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer {}'.format(self.api_key),
        }
        response = requests.request("GET", url, headers=headers, data=payload)
        id_list = []
        try:
            for i in response.json()['data']:
                id_list.append(i['id'])
        except:
            raise Exception(response.json())
        return id_list
    