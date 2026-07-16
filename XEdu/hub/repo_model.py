# -*- coding: utf-8 -*-
import os,sys
import importlib.util
import onnxruntime as ort

class RepoModel:
    def __init__(self, download_path, repo):
        self.repo = repo
        url = os.path.abspath(download_path)
        if os.path.exists(url):
            self.url = os.path.abspath(url)
            self.local = True
        else:
            self.url = url
            self.local = False
        self.preprocess = None
        self.postprocess = None
        self.model = None
        self.custom_inference = None

    def load_local_repo(self):
        assert os.path.exists(os.path.join(self.url,self.repo,"data_process.py")), "data_process.py not found."
        assert os.path.exists(os.path.join(self.url,self.repo,"model.onnx")), "model.onnx not found."

        process = os.path.join(self.url,self.repo,"data_process.py")
        spec = importlib.util.spec_from_file_location("data_process", process)
        data_process = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_process)
        if hasattr(data_process,"inference"):
            self.custom_inference = data_process.inference

        if  hasattr(data_process,"preprocess"):
            self.preprocess = data_process.preprocess
        if hasattr(data_process,"postprocess"):
            self.postprocess = data_process.postprocess
        checkpoint = os.path.join(self.url,self.repo,"model.onnx")
        self.model = ort.InferenceSession(checkpoint, None)

        # 获得模型输入输出名称
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        
    def load_modelscope_repo(self, local_path):
        try: 
            from modelscope import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "ModelScope support requires the optional dependency 'modelscope'. "
                "Install it before using repo-based remote models."
            ) from exc
        model_dir = snapshot_download(self.repo,local_dir=os.path.join(local_path,self.repo))
        process = os.path.join(model_dir,"data_process.py")
        spec = importlib.util.spec_from_file_location("data_process", process)
        data_process = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_process)
        if hasattr(data_process,"inference"):
            self.custom_inference = data_process.inference

        if  hasattr(data_process,"preprocess"):
            self.preprocess = data_process.preprocess
        if hasattr(data_process,"postprocess"):
            self.postprocess = data_process.postprocess
        checkpoint = os.path.join(model_dir,"model.onnx")
        self.model = ort.InferenceSession(checkpoint, None)
        # 获得模型输入输出名称
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        
    def inference(self, data, **kwarg):
        if self.custom_inference is not None:
            return self._custom_infer(data, **kwarg)
        else:
            return self._pre_post_infer(data,self.input_name,self.output_name)

    def _pre_post_infer(self, data,input_name='input',output_name='output'):
        if self.preprocess is not None:
            data = self.preprocess(data)
        ort_inputs = {input_name: data}
        self.repo_res = self.model.run([output_name], ort_inputs)
        # print(self.custom_res[0].shape)
        if self.postprocess is not None:
            self.repo_res = self.postprocess(self.repo_res)

        return self.repo_res

    def _custom_infer(self, data, **kwarg):
        if self.custom_inference is None:
            raise RuntimeError("Custom inference function not available")
        self.custom_res = self.custom_inference(data, **kwarg)
        return self.custom_res
