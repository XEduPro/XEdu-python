# -*- encoding: utf-8 -*-
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, Union,Iterable, Iterator, TypeVar
import cv2
import json
import numpy as np
import warnings
import ast
if TYPE_CHECKING:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
import onnxruntime as ort
import time
import io
import matplotlib.pyplot as plt
import pprint
import os
import shutil
from PIL import Image
from .tokenizer.qa_tokenizer import *
from .tokenizer.qa_squadprocess import *
from .tokenizer.clip_tokenizer import Preprocessor, Tokenizer
from .datatset_class import coco_class,imagenet_class
import copy
import requests
from tqdm import tqdm
from .models import efficientVIT_sam as sam
from .errorcode import ErrorCodeFactory as ecf
T = TypeVar("T")

def _xedu_home_dir() -> str:
    custom_home = os.environ.get("XEDU_HOME")
    if custom_home:
        return os.path.abspath(os.path.expanduser(custom_home))
    if os.name == "nt":
        base_dir = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or os.path.expanduser("~")
        )
    else:
        base_dir = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
    return os.path.join(os.path.abspath(base_dir), "XEdu")

def _default_hub_dir(*parts: str) -> str:
    return os.path.join(_xedu_home_dir(), "hub", *parts)

def _resolve_download_dir(download_path: Optional[str], default_subdir: str) -> str:
    if download_path is None:
        return _default_hub_dir(default_subdir)
    return os.path.abspath(os.path.expanduser(download_path))

def _migrate_legacy_file(target_path: str, *legacy_dir_names: str) -> None:
    if os.path.exists(target_path):
        return
    file_name = os.path.basename(target_path)
    for legacy_dir_name in legacy_dir_names:
        legacy_path = os.path.join(os.getcwd(), legacy_dir_name, file_name)
        if os.path.exists(legacy_path):
            os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
            shutil.copy2(legacy_path, target_path)
            return

def _migrate_legacy_directory(target_dir: str, *legacy_dir_names: str) -> None:
    if os.path.exists(target_dir):
        return
    for legacy_dir_name in legacy_dir_names:
        legacy_dir = os.path.join(os.getcwd(), legacy_dir_name)
        if os.path.isdir(legacy_dir):
            shutil.copytree(legacy_dir, target_dir)
            return

def to_batches(items: Iterable[T], size: int) -> Iterator[List[T]]:
    """
    Splits an iterable (e.g. a list) into batches of length `size`. Includes
    the last, potentially shorter batch.

    Examples:
        >>> list(to_batches([1, 2, 3, 4], size=2))
        [[1, 2], [3, 4]]
        >>> list(to_batches([1, 2, 3, 4, 5], size=2))
        [[1, 2], [3, 4], [5]]

    Args:
        items: The iterable to split.
        size: How many elements per batch.
    """
    if size < 1:
        raise ValueError("Chunk size must be positive.")

    batch = []
    for item in items:
        batch.append(item)

        if len(batch) == size:
            yield batch
            batch = []

    # The last, potentially incomplete batch
    if batch:
        yield batch



task_dict = {
        "pose_body17":"body17.onnx",
        "pose_body17_l":"body17_l.onnx",
        "pose_body26":"body26.onnx",
        "pose_wholebody133":"whole133.onnx",
        "pose_face106":"face106.onnx",
        "pose_hand21":"hand21.onnx",
        "det_body":"bodydetect.onnx",
        "det_body_l":"bodydetect_l.onnx",
        "det_coco":"cocodetect.onnx",
        "det_coco_l":"cocodetect_l.onnx",
        "det_hand":"handdetect.onnx",
        "cls_imagenet":"imagenet1k.onnx",
        "gen_style":"gen_style_mosaic.onnx",
        "gen_style_mosaic":"gen_style_mosaic.onnx",
        "gen_style_candy":"gen_style_candy.onnx",
        "gen_style_rain-princess":"gen_style_rain-princess.onnx",
        "gen_style_udnie":"gen_style_udnie.onnx",
        "gen_style_pointilism":"gen_style_pointilism.onnx",
        "gen_style_custom":"gen_style_custom.onnx",
        "nlp_qa":"nlp_qa.onnx",
        "drive_perception":"drive_perception.onnx",
        "embedding_image":"embedding_image.onnx",
        "embedding_text":"embedding_text.onnx",
        "embedding_audio":"embedding_audio.onnx",
        "gen_color":"gen_color.onnx",
        "segment_anything":['seg_sam_encoder.onnx','seg_sam_decoder.onnx'],
        "depth_anything":"depth_anything.onnx",
        "det_face":"face_detection_yunet_2023mar.onnx",
        "ocr":"...",
        "mmedu":"...",
        "basenn":"...",
        "baseml":"...",
        "custom":"...",
}
style_list = ['mosaic','candy','rain-princess','udnie','pointilism']
FACE_DET_LEGACY_PARAMS = {"scaleFactor", "minNeighbors", "minSize", "maxSize"}

class Downloader(object):
    def __init__(self, url, file_path,model_name=None,output_dir='checkpoints',overwrite=False):
        self.url = url
        self.file_path = os.path.abspath(os.path.expanduser(file_path))
        if model_name is None:
            self.model_name = os.path.basename(self.file_path)
        else:
            self.model_name = model_name
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))

    def start(self):
        os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
        res_length = requests.get(self.url, stream=True)
        total_size = int(res_length.headers['Content-Length'])
        if os.path.exists(self.file_path):
            temp_size = os.path.getsize(self.file_path)
        else:
            temp_size = 0
        print(f"model_repo:xedu/hub-model, model_name:{self.model_name}, output:{self.output_dir}, overwrite:False")
        headers = {'Range': 'bytes=%d-' % temp_size,
                   "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:81.0) Gecko/20100101 Firefox/81.0"}
        res_left = requests.get(self.url, stream=True, headers=headers)
        total_size_in_bytes = int(res_left.headers['Content-Length'])
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(self.file_path, "ab") as f:
            for chunk in res_left.iter_content(chunk_size=1024):
                temp_size += len(chunk)
                progress_bar.update(len(chunk))
                f.write(chunk)
                
        progress_bar.close()
        print(f"download model repo:xedu/hub-model, file_name:{self.model_name} ")

class Workflow:
    """
        Workflow类用于加载预训练模型以解决各类任务。
        目前支持的任务有：
            - pose_body17：人体关键点检测，17个关键点
            - pose_body17_l：人体关键点检测，17个关键点，模型更大
            - pose_body26：人体关键点检测，26个关键点
            - pose_wholebody133：全身关键点检测，包括人体、人脸和手部共133个关键点
            - pose_face106：人脸关键点检测，106个关键点
            - pose_hand21：手部关键点检测，21个关键点
            - det_body：人体检测
            - det_body_l：人体检测，模型更大
            - det_coco：物体检测，80类，基于COCO数据集
            - det_coco_l：物体检测，基于COCO数据集，模型更大
            - det_hand：手部检测
            - det_face：人脸检测
            - cls_imagenet：图像分类，1000类，基于ImageNet数据集
            - gen_style：风格迁移，5种风格
            - gen_color：图像着色
            - nlp_qa：问答系统，基于SQuAD数据集
            - drive_perception：全景驾驶感知系统，包括交通对象检测、可行驶道路区域分割和车道检测任务
            - embedding_image：CLIP图像嵌入
            - embedding_text：CLIP文本嵌入   
            - embedding_audio：CLAP音频嵌入
            - ocr：光学字符识别，基于rapidocr
            - segment_anything：图像分割，基于Segment Anything Model (SAM)
            - depth_anything：深度估计
            - mmedu：MMEdu模型推理
            - basenn：BaseNN模型推理
            - baseml：BaseML模型推理


        Attributes:
            task：任务类型，可选范围如以上列出。    
            checkpoint：模型权重文件的路径。
            download_path：模型文件即将下载到的路径。

        更多用法及算法详解请参考：https://xedu.readthedocs.io/zh-cn/master/xedu_hub/introduction.html
    """
    @classmethod
    def support_task(cls):
        return list(task_dict.keys())
    @classmethod
    def _task_dict(cls):
        return task_dict
    @classmethod
    def coco_class(cls):
        return coco_class
    @classmethod
    def support_style(cls):
        return style_list

    def __init__(self, task=None,checkpoint=None,download_path=None,repo=None,**kwargs):
        if repo is not None:
            path = _resolve_download_dir(download_path, "repo")
            self.repo = repo
            from .repo_model import RepoModel
            self.model = RepoModel(path,repo)
            _migrate_legacy_directory(os.path.join(path, repo), os.path.join("repo", repo))

            if os.path.exists(os.path.join(path,repo)):
                self.model.load_local_repo()
            else:
                self.model.load_modelscope_repo(path)
            return 
        self.repo = None
        self.demo_input = None
        self.input_shape = None
        task = task.lower()
        self.task_dict = task_dict
        self.task_nick_name ={
            'body':"pose_body17",
            'body_l':"pose_body17_l",
            'pose_body_l':"pose_body17_l",
            "face":"pose_face106",
            "hand":"pose_hand21",
            "wholebody":"pose_wholebody133",
            "face106":"pose_face106",
            "hand21":"pose_hand21",
            "wholebody133":"pose_wholebody133",
            'body17':"pose_body17",
            'body26':"pose_body26",
            'pose_hand':"pose_hand21",
            'pose_body':"pose_body17",
            "pose_wholebody":"pose_wholebody133",
            "pose_face":"pose_face106",
            "handdetect":"det_hand",
            "cocodetect":"det_coco",
            "facedetect":"det_face",
            "bodydetect":"det_body",
        }
        path = _resolve_download_dir(download_path, "checkpoints")
        self.download_path = path
        self.asset_root = os.path.dirname(path)
        if task not in self.task_dict.keys() and task not in self.task_nick_name.keys():
            raise ValueError(f"Error Code: -310. No such task: '{task}'. Please refer to 'support_task()' method for a list of supported tasks. ")
        if task in self.task_nick_name.keys():
            self.task = self.task_nick_name[task]
        else:
            self.task = task
        self.color_dict = {
            'red':[255,0,0],
            'orange':[255,125,0],
            'yellow':[255,255,0],
            'green':[0,255,0],
            'blue':[0,0,255],
            'purple':[255,0,255],
            'l_red':[128,0,0],
            'l_orange':[128,64,0],
            'l_yellow':[128,128,0],
            'l_green':[0,128,0],
            'l_blue':[0,0,128],
            'l_purple':[128,0,128],
        }
        if self.task == 'nlp_qa':
            self.answers = []
            self.questions = []
            self.contexts = []
            self.doc = None
        default_task = ['embedding_audio','depth_anything','det_body','det_body_l','det_coco','det_coco_l','pose_body17','pose_body17_l','pose_body26','pose_wholebody133','pose_face106','pose_hand21','det_hand','cls_imagenet','gen_style','nlp_qa','drive_perception','embedding_image','embedding_text','gen_color']
        
        if checkpoint is None and self.task in default_task: # 若不指定权重文件，则使用对应任务的默认模型
            if self.task == 'gen_style':
                self.style = kwargs.get('style',None)
                if isinstance(self.style,int): # 按照索引选择风格
                    style_map = ['mosaic','candy','rain-princess','udnie','pointilism']
                    self.style = style_map[self.style]
                elif isinstance(self.style,str) and self.style not in ['mosaic','candy','rain-princess','udnie','pointilism']: # 自定义风格
                    self.style_img = self.style
                    self.style = 'custom'
                if self.style is  None:
                    self.style = 'mosaic'
                self.task = 'gen_style_{}'.format(self.style)
                checkpoint = "gen_style_{}.onnx".format(self.style)
            else:
                checkpoint = self.task_dict[self.task]
            checkpoint = os.path.join(path, checkpoint)
            _migrate_legacy_file(checkpoint, "checkpoint", "checkpoints")
            if not os.path.exists(checkpoint): # 本地未检测到模型，云端下载默认模型
                if self.task == 'pose_face106':
                    raise RuntimeError(
                        "pose_face106 模型暂不支持自动下载：历史版本在此处错误地复用了 "
                        "pose_wholebody133 的下载地址，会导致下载到错误的模型文件，因此改为"
                        "显式报错而不是静默使用错误模型。请手动指定本地模型文件，例如："
                        "wf(task='pose_face106', checkpoint='/path/to/face106.onnx')"
                    )
                print("本地未检测到{}任务对应模型，云端下载中...".format(self.task))
                os.makedirs(path, exist_ok=True)

                baseurl='https://www.openinnolab.org.cn/'
                model_name_map_download ={
                    'cls_imagenet':'/res/api/v1/file/creator/09a4c4f4-7034-45a5-a0c1-a747da5a2766.onnx&name=cls_imagenet.onnx',    
                    'det_body':'/res/api/v1/file/creator/8137e4ca-482d-48fa-b57f-bfa50f7768be.onnx&name=det_body.onnx',
                    'det_body_l':'/res/api/v1/file/creator/d6d5680e-b3ef-4624-9a9f-52f1892f0045.onnx&name=det_body_l.onnx',
                    'det_coco':'/res/api/v1/file/creator/e4c39ead-ff3b-4810-ab4d-a8f3757ff1bb.onnx&name=det_coco.onnx',
                    'det_coco_l':'/res/api/v1/file/creator/8be89312-4ff7-4dc7-ba19-7d759d8e713a.onnx&name=det_coco_l.onnx',
                    'det_hand':'/res/api/v1/file/creator/6172ad06-8a97-4d47-bcc3-cdbdda0c0187.onnx&name=det_hand.onnx',
                    
                    'pose_body17':'/res/api/v1/file/creator/b94f252e-03de-4491-b9f3-042c57c7671f.onnx&name=pose_body17.onnx',
                    'pose_body17_l':'/res/api/v1/file/creator/8e71a720-e87b-42e7-8498-8a6d07473941.onnx&name=pose_body17_l.onnx',
                    'pose_body26':'/res/api/v1/file/creator/2de9dd14-93c7-4b89-ac79-da3231c79d01.onnx&name=pose_body26.onnx',
                    'pose_wholebody133':'/res/api/v1/file/creator/98e010a3-76f4-4209-bba9-33fba2fe1281.onnx&name=pose_wholebody133.onnx',
                    'pose_hand21':'/res/api/v1/file/creator/e5e5540b-3475-42f8-be0b-6ea8d46d577b.onnx&name=pose_hand21.onnx',
                    # pose_face106 故意不在此列出：见上方对 self.task == 'pose_face106' 的
                    # 显式 RuntimeError 短路检查，原因是原上游代码在此处错误复用了
                    # pose_wholebody133 的下载地址。

                    'embedding_image':'/res/api/v1/file/creator/69aebb8e-3202-4022-9618-a64560ffef76.onnx&name=embedding_image.onnx',
                    'embedding_text':'/res/api/v1/file/creator/ce38d2ad-e8be-4e6a-990a-a6d818e5655b.onnx&name=embedding_text.onnx',
                    'embedding_audio':'/res/api/v1/file/creator/3fb25823-aeb7-4866-9617-937f5079af4a.onnx&name=embedding_audio.onnx',
                    
                    'gen_color':'/res/api/v1/file/creator/733caa05-0357-4e52-a7b0-ce9a9419f959.onnx&name=gen_color.onnx',
                    'gen_style_candy':'/res/api/v1/file/creator/bc24e059-131a-49d0-b663-45289156bbc9.onnx&name=gen_style_candy.onnx',
                    'gen_style_mosaic':'/res/api/v1/file/creator/965b190c-6008-43dd-a037-94a99e55f78a.onnx&name=gen_style_mosaic.onnx',
                    'gen_style_pointilism':'/res/api/v1/file/creator/9e5fb84f-fcd5-497f-a59f-9359e430e549.onnx&name=gen_style_pointilism.onnx',
                    'gen_style_rain-princess':'/res/api/v1/file/creator/f193af6e-8eaf-43c1-913c-85b226da4e47.onnx&name=gen_style_rain-princess.onnx',
                    'gen_style_udnie':'/res/api/v1/file/creator/3691c4c2-877b-4137-b621-7eb7dede54e3.onnx&name=gen_style_udnie.onnx',
                    'gen_style_custom':'/res/api/v1/file/creator/17b8f5e6-94b6-44a3-b15b-486f6fc6a142.onnx&name=gen_style_custom.onnx',
                                        
                    'drive_perception':'/res/api/v1/file/creator/add78652-51c0-41e3-ab56-a52dd7374e54.onnx&name=drive_perception.onnx',
                    
                    'nlp_qa':'/res/api/v1/file/creator/b1938fd0-6ffa-4c91-a98e-170cd3b1c520.onnx&name=nlp_qa.onnx',
                    'depth_anything':'/res/api/v1/file/creator/ffa0880a-4900-4ef5-8106-562eb14e7e8e.onnx&name=depth_anything.onnx'
                }
                for key in model_name_map_download.keys():
                    model_name_map_download[key] = baseurl + model_name_map_download[key]
                downloader = Downloader(model_name_map_download[self.task], os.path.join(path, self.task_dict[self.task]),output_dir=path)
                downloader.start()

            if self.task == 'embedding_audio':
                from .models.clap import CLAP
                self.model = CLAP(model_path=checkpoint)
            else:
                self.model = ort.InferenceSession(checkpoint, None)
        elif self.task =="det_face":
                checkpoint = checkpoint or os.path.join(path, self.task_dict[self.task])
                if checkpoint.lower().endswith(".xml"):
                    self.model = cv2.CascadeClassifier(checkpoint)
                else:
                    _migrate_legacy_file(checkpoint, "checkpoint", "checkpoints")
                    detector_url = (
                        "https://github.com/opencv/opencv_zoo/raw/main/models/"
                        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
                    )
                    os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
                    if not os.path.exists(checkpoint):
                        print("本地未检测到{}任务对应模型，云端下载中...".format(self.task))
                        downloader = Downloader(
                            detector_url,
                            checkpoint,
                            output_dir=os.path.dirname(checkpoint) or path,
                        )
                        downloader.start()
                    try:
                        self.model = cv2.FaceDetectorYN_create(
                            checkpoint, "", (320, 320), 0.6, 0.3, 5000
                        )
                    except cv2.error:
                        if os.path.exists(checkpoint):
                            os.remove(checkpoint)
                        print("det_face模型文件损坏，重新下载中...")
                        downloader = Downloader(
                            detector_url,
                            checkpoint,
                            output_dir=os.path.dirname(checkpoint) or path,
                        )
                        downloader.start()
                        self.model = cv2.FaceDetectorYN_create(
                            checkpoint, "", (320, 320), 0.6, 0.3, 5000
                        )
        elif self.task == "ocr":
                try :
                    from rapidocr_onnxruntime import RapidOCR
                except ImportError as exc:
                    raise ImportError(
                        "The 'ocr' task requires the optional dependency "
                        "'rapidocr_onnxruntime'. Install it before using OCR."
                    ) from exc
                self.model = RapidOCR(text_score=0.2)
        elif self.task.lower() == "mmedu":
            if checkpoint is None:
                print("请先指定通过MMEdu导出的onnx模型路径。")
                return
            assert os.path.exists(checkpoint),ecf.NO_SUCH_CHECKPOINT(checkpoint)
            assert os.path.splitext(checkpoint)[-1]==".onnx",ecf.CHECKPOINT_TYPE(os.path.splitext(checkpoint)[-1].strip("."))
            from .BaseDeploy import _BaseDeploy as bd
            self.model = bd(checkpoint)
        elif self.task == 'baseml':
            if checkpoint is None:
                print("请先指定通过BaseML导出的pkl模型路径。")
                return
            assert os.path.exists(checkpoint),ecf.NO_SUCH_CHECKPOINT(checkpoint)
            assert os.path.splitext(checkpoint)[-1]==".pkl",ecf.CHECKPOINT_TYPE(os.path.splitext(checkpoint)[-1].strip("."))

            try:
                import sklearn
                import joblib
            except ImportError as exc:
                raise ImportError(
                    "The 'baseml' task requires the optional dependency "
                    "'scikit-learn'. Install it before using BaseML checkpoints."
                ) from exc
            model = joblib.load(checkpoint)
            if isinstance(model, dict):
                self.model = model['model']
                try:
                    self.demo_input = model['demo_input']
                    self.input_shape = model['input_shape']
                except:
                    pass
            else:
                self.model = model
        elif  self.task == 'segment_anything':
            if checkpoint is None:
                checkpoint = [os.path.join(path, task_dict[self.task][0]),os.path.join(path, task_dict[self.task][1])]
            else:
                assert isinstance(checkpoint,list) and len(checkpoint)==2, "checkpoint should be a list of two paths for encoder and decoder."
            decoder_url = 'https://www.openinnolab.org.cn/res/api/v1/file/creator/70f02a96-6998-4196-92ac-c61a9a841c66.onnx&name=seg_sam_decoder.onnx'
            encoder_url  ='https://www.openinnolab.org.cn//res/api/v1/file/creator/b0baaf01-8673-4762-a99b-f47661454395.onnx&name=seg_sam_encoder.onnx'
            _migrate_legacy_file(checkpoint[0], "checkpoint", "checkpoints")
            _migrate_legacy_file(checkpoint[1], "checkpoint", "checkpoints")
            if not os.path.exists(checkpoint[0]):
                print("本地未检测到{}任务对应encoder模型，云端下载中...".format(self.task))
                if not os.path.exists(path):
                    os.mkdir(path)
                downloader = Downloader(encoder_url, checkpoint[0],output_dir=path)
                downloader.start()
            if not os.path.exists(checkpoint[1]):
                print("本地未检测到{}任务对应decoder模型，云端下载中...".format(self.task))
                downloader = Downloader(decoder_url, checkpoint[1],output_dir=path)
                downloader.start()
            self.model = [sam.SamEncoder(model_path=checkpoint[0]),sam.SamDecoder(model_path=checkpoint[1])]
        else:
            assert os.path.exists(checkpoint),ecf.NO_SUCH_CHECKPOINT(checkpoint)
            assert os.path.splitext(checkpoint)[-1]==".onnx",ecf.CHECKPOINT_TYPE(os.path.splitext(checkpoint)[-1].strip("."))
            self.model = ort.InferenceSession(checkpoint, None)
        if self.task[:3] in ['det','cls'] and 'face' not in self.task:
            self._check_mmedu(self.model)
        print(f"{self.task}任务模型加载成功！")
    
    def _check_mmedu(self,model):
        model_meta = model.get_modelmeta()
        key='MODEL_INFO'
        if key in model_meta.custom_metadata_map:
            unicode_string = model_meta.custom_metadata_map[key]
            # 不匹配的任务名
            info = f'Error code: -311. Input task type "{self.task}" does not match the model which is generated by MMEdu. Please set task="mmedu" instead of "{self.task}".'
            raise ValueError(info)
            # print(f'Model info: {unicode_string[:100] + "..." if len(unicode_string) > 100 else unicode_string}')
            

    def inference(self,data=None,show=False,img_type=None,thr=0.3,bbox=None,target_class=None,erase=False,
                  preprocess=None, postprocess=None,**kwargs):
        self._get_img = img_type
        self.erase=erase
        if self.repo is not None:
            return self.model.inference(data,**kwargs)
        if isinstance(data,str) and self.task not in ['nlp_qa','embedding_text','embedding_audio']:
            assert os.path.exists(data),ecf.NO_SUCH_FILE(data)
            filetype = ['.jpg','.png','.jpeg','.jiff']
            assert os.path.splitext(data.lower())[-1] in filetype,ecf.INFER_DATA_TYPE(os.path.splitext(data)[-1].strip("."))


        if self.task in ["pose_body17","pose_body26","pose_face106","pose_hand21","pose_wholebody133","pose_body17_l"]:
            return self._pose_infer(data,show,img_type,bbox)
        elif self.task in['det_body','det_coco','det_hand','det_body_l','det_coco_l']:
            return self._det_infer(data,show,img_type,thr,target_class)
        elif self.task in ['det_face']:
            return self._face_det_infer(data,show,img_type,**kwargs)
        elif self.task in ['cls_imagenet']:
            return self._cls_infer(data,show, img_type)
        elif self.task in ['gen_style_mosaic','gen_style_candy','gen_style_rain-princess','gen_style_udnie','gen_style_pointilism','gen_style_custom']:
            if self.style in ['mosaic','candy','rain-princess','udnie','pointilism'] or isinstance(self.style,int): # 预设风格
                return self._style_infer(data,show,img_type)
            else: # 自定义风格
                return self._style_custom_infer(data,show,img_type)
        elif self.task in ['ocr']:
            return self._ocr_infer(data,show,img_type)
        elif self.task in ['mmedu']:
            return self._mmedu_infer(data,show,img_type,thr)
        elif self.task in ['basenn']:
            self.ix2word = None
            return self._basenn_infer(data,show,img_type)
        elif self.task in ['baseml']:
            return self._baseml_infer(data,show,img_type)
        elif self.task in ['custom']:
            return self._custom_infer(data, preprocess, postprocess)
        elif self.task in ['nlp_qa']:
            return self._nlp_qa_infer(data,kwargs.get('context',None))
        elif self.task in ['drive_perception']:
            return self._drive_perception_infer(data,show,img_type,thr)
        elif self.task in ['embedding_image']:
            if isinstance(data, str): # 单张图片
                data = [Image.open(data).convert("RGB")]
            elif isinstance(data, Iterable) and all(isinstance(item, str) for item in data): # 多张图片
                data = [Image.open(item).convert("RGB") for item in data]
            elif isinstance(data, np.ndarray): 
                if len(data.shape) == 3: 
                    data = np.expand_dims(data, axis=0)
                if data.shape[-1] != 3 and data.shape[-3] == 3:
                    data = np.transpose(data, (0, 2, 3, 1))
            elif isinstance(data,Image.Image): # 单张图片
                data = [data.convert("RGB")]
            self._preprocessor = Preprocessor()
            self._batch_size = 16
            return self._embedding_image_infer(data)
        elif self.task in ['embedding_text']:
            if isinstance(data, str): # 单个文本
                data = [data]
            self._tokenizer = Tokenizer()
            self._batch_size = 16
            return self._embedding_text_infer(data)
        elif self.task in ['gen_color']:
            return self._gen_color_infer(data,show,img_type)
        elif self.task in ['segment_anything']:
            return self._seg_sam_infer(data,show,img_type,**kwargs)
        elif self.task in ['depth_anything']:
            return self._mde_da_infer(data,show,img_type)
        elif self.task in ['embedding_audio']:
            return self._embedding_audio_infer(data)
        else:
            raise NotImplementedError

    def _embedding_audio_infer(self,data):
        if isinstance(data,str):
            assert os.path.exists(data),ecf.NO_SUCH_FILE(data)
            data = [data]
        res = self.model.get_audio_embedding(data)
        return res 

    def _mde_da_infer(self,data,show,img_type):
        self._get_img = 'cv2'
        if isinstance(data,str):
            assert os.path.exists(data),ecf.NO_SUCH_FILE(data)
            np_data = cv2.imread(data)
        else:
            np_data = data

        from .models.depth_anything import da_onnx_infer
        self.depth_res = da_onnx_infer(np_data,self.model)
        if show:
            self.show(self.depth_res)
        if img_type:
            return self.depth_res,self.depth_res
        return self.depth_res

                

    def _seg_sam_infer(self,data,show,img_type,mode='point',prompt=None):
        if isinstance(data,str):
            assert os.path.exists(data),ecf.NO_SUCH_FILE(data)
            np_data = cv2.imread(data)
        else:
            np_data = data

        raw_img = cv2.cvtColor(np_data, cv2.COLOR_BGR2RGB)
        origin_image_size = raw_img.shape[:2]
        img = sam.preprocess(raw_img, img_size=512)

        img_embeddings = self.model[0](img) # encoder
        if mode == "point":
            if prompt is None:
                prompt = np.array([[origin_image_size[1] /2,origin_image_size[0]/2]],dtype='float32')
            else:
                if len(np.array(prompt).shape) == 1:
                    prompt = np.array(prompt)[None,:]
                else:
                    prompt = np.array(prompt)
            ax = [[1] for i in range(len(prompt))]
            point = np.concatenate([prompt, ax],axis=1)
            point = np.array([point],dtype='float32')
            point_coords = point[..., :2]
            point_labels = point[..., 2]
            masks, self.mask_scores, _ = self.model[1].run(
                img_embeddings=img_embeddings,
                origin_image_size=origin_image_size,
                point_coords=point_coords,
                point_labels=point_labels,
            ) # decoder
            self.masks = masks[0]
        elif mode == "box":
            if prompt is None:
                boxes = np.array([[0,0,origin_image_size[1],origin_image_size[0]]],dtype='float32')
            else:
                if len(np.array(prompt).shape) == 1:
                    boxes = np.array(prompt)[None,:]
                else:
                    boxes = np.array(prompt)
            masks, self.mask_scores, _ = self.model[1].run(
                img_embeddings=img_embeddings,
                origin_image_size=origin_image_size,
                boxes=boxes,
            ) # decoder
            self.masks = masks[0]
        else:
            raise NotImplementedError
        if img_type is None:
            self._get_img = "pil"
        else:
            self._get_img = img_type
            # 增加alpha通道
            # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2BGRA)
            w, h, _ = raw_img.shape

            fig = plt.figure(figsize=(h/100,w/100), dpi=100)
            plt.imshow(raw_img)
            if mode == "point":
                for mask in masks[0]:
                    sam.show_mask(mask, plt.gca(), random_color=len(masks) > 1)
                sam.show_points(point_coords, point_labels, plt.gca())
            elif mode == "box":
                for mask in masks[0]:
                    sam.show_mask(mask, plt.gca(), random_color=len(masks) > 1)
                for box in boxes:
                    sam.show_box(box, plt.gca())

            plt.axis("off")
            plt.tight_layout()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            # 将画布保存到字节流
            buf = io.BytesIO()
            plt.savefig(buf, dpi=fig.dpi, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)

            # 使用PIL从字节流中加载图像并转换为数组
            img = Image.open(buf)
            return_img = np.array(img)
            if img_type == 'cv2':
                return_img = cv2.cvtColor(return_img,cv2.COLOR_RGB2BGR)
            plt.close(fig)
            if show:
                # fig = plt.figure(figsize=(10,10))
                self.show(return_img)
            # print("data",img.shape)
            return masks[0], return_img
        return masks[0]
        
    def _gen_color_infer(self,data=None,show=False,img_type=None):

        def load_img(img_path):
            if isinstance(img_path,str):
                out_np = np.asarray(Image.open(img_path))
            else:
                out_np = np.asarray(img_path)
            if(out_np.ndim==2):
                out_np = np.tile(out_np[:,:,None],3)
            return out_np

        def resize_img(img, HW=(256,256), resample=3):
            return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

        def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
            img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

            img_rgb_orig = np.float32(img_rgb_orig * (1./255))
            img_lab_orig = cv2.cvtColor(img_rgb_orig, cv2.COLOR_RGB2LAB)

            img_rgb_rs = np.float32(img_rgb_rs * (1./255))
            img_lab_rs = cv2.cvtColor(img_rgb_rs, cv2.COLOR_BGR2LAB)

            img_l_orig = img_lab_orig[:,:,0]
            img_l_rs = img_lab_rs[:,:,0]

            tens_orig_l = img_l_orig[None,None,:,:]
            tens_rs_l = img_l_rs[None,None,:,:]

            return (tens_orig_l, tens_rs_l)

        def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
            HW_orig = tens_orig_l.shape[2:]
            HW = out_ab.shape[2:]

            if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
                # out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
                out_a = cv2.resize(out_ab[0,0,:,:], (HW_orig[1], HW_orig[0]), interpolation=cv2.INTER_LINEAR)
                out_b = cv2.resize(out_ab[0,1,:,:], (HW_orig[1], HW_orig[0]), interpolation=cv2.INTER_LINEAR)
                out_ab_orig = np.concatenate(([out_a],[out_b]),axis=0)
                out_ab_orig = out_ab_orig[None,...]
            else:
                out_ab_orig = out_ab

            out_lab_orig = np.concatenate((tens_orig_l, out_ab_orig), axis=1)

            return cv2.cvtColor(out_lab_orig[0,...].transpose((1,2,0)), cv2.COLOR_LAB2RGB)

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        img = load_img(data)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
        input_data = tens_l_rs
        output_data = self.model.run([output_name], {input_name: input_data})[0]
        self.color_res = postprocess_tens(tens_l_orig, output_data)
        if img_type:
            self._get_img = img_type
            if img_type =='cv2':
                # image = cv2.cvtcolor(image,cv2.BGR2RGB)
                self.color_res = cv2.cvtColor(self.color_res,cv2.COLOR_BGR2RGB)
            if show:
                self.show(self.color_res)
            return self.color_res,self.color_res
        else:
            self._get_img = 'pil'
        return self.color_res

    def _empty_embedding(self):
        return np.empty((0, 512), dtype=np.float32)

    def _embedding_image_infer(self,images: Iterable[Union[Image.Image, np.ndarray]],with_batching: bool = True,):
        """Compute the embeddings for a list of images.

        Args:
            images: A list of images to run on. Each image must be a 3-channel
                (RGB) image. Can be any size, as the preprocessing step will
                resize each image to size (224, 224).
            with_batching: Whether to use batching 

        Returns:
            An array of embeddings of shape (len(images), embedding_size).
        """
        if not with_batching or self._batch_size is None:
            images = [self._preprocessor.encode_image(image) for image in images]
            if not images:
                return self._empty_embedding()

            batch = np.concatenate(images)

            return self.model.run(None, {"IMAGE": batch})[0]

        else:
            embeddings = []
            for batch in to_batches(images, self._batch_size):
                embeddings.append(
                    self._embedding_image_infer(batch, with_batching=False)
                )

            if not embeddings:
                return self._empty_embedding()

            return np.concatenate(embeddings)

    def _embedding_text_infer(self,texts: Union[Iterable[str],str], with_batching: bool = True)->np.array:
        """Compute the embeddings for a list of texts.

        Args:
            texts: A list of texts to run on. Each entry can be at most
                77 characters.
            with_batching: Whether to use batching - see the `batch_size` param
                in `__init__()`

        Returns:
            An array of embeddings of shape (len(texts), embedding_size).
        """
        if not with_batching or self._batch_size is None:
            text = self._tokenizer.encode_text(texts)
            if len(text) == 0:
                return self._empty_embedding()

            return self.model.run(None, {"TEXT": text})[0]
        else:
            embeddings = []
            for batch in to_batches(texts, self._batch_size):
                embeddings.append(
                    self._embedding_text_infer(batch, with_batching=False)
                )

            if not embeddings:
                return self._empty_embedding()

            return np.concatenate(embeddings)    
        
    def _drive_perception_infer(self,data=None,show=False,img_type=None,thr=0.3):
        model_inputs = self.model.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])
        self.confThreshold = thr

        if isinstance(data, str):
            image = cv2.imread(data)
        else:
            image = data
        image_width, image_height = image.shape[1], image.shape[0]
        ratioh = image_height / self.input_height
        ratiow = image_width / self.input_width

        input_image = cv2.resize(image, dsize=(self.input_width, self.input_height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')
        input_image = input_image / 255.0

        results = self.model.run(None, {self.input_name: input_image})
        scores_ = results[2]
        batchno_classid_y1x1y2x2 = results[3]
        # Traffic Object Detection
        bboxes, scores, class_ids = [], [], []
        for score, batchno_classid_y1x1y2x2_ in zip(scores_, batchno_classid_y1x1y2x2):
            score = float(np.asarray(score).reshape(-1)[0])
            if score < self.confThreshold:
                continue

            class_id = int(batchno_classid_y1x1y2x2_[1])
            y1 = batchno_classid_y1x1y2x2_[2]
            x1 = batchno_classid_y1x1y2x2_[3]
            y2 = batchno_classid_y1x1y2x2_[4]
            x2 = batchno_classid_y1x1y2x2_[5]
            y1 = int(y1 * ratioh)
            x1 = int(x1 * ratiow)
            y2 = int(y2 * ratioh)
            x2 = int(x2 * ratiow)

            bboxes.append([x1, y1, x2, y2])
            class_ids.append(class_id)
            scores.append(score)

        # Drivable Area Segmentation
        drivable_area = np.squeeze(results[0], axis=0)
        mask = np.argmax(drivable_area, axis=0).astype(np.uint8)
        area_mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

        # Lane Line
        lane_line = np.squeeze(results[1])
        mask = np.where(lane_line > 0.5, 1, 0).astype(np.uint8)
        lane_mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        # 将x1y1x2y2转化为x1y1w,h
        if len(bboxes) != 0:
            bboxes = np.array(bboxes)
            bboxes[:,2] = bboxes[:,2] - bboxes[:,0]
            bboxes[:,3] = bboxes[:,3] - bboxes[:,1]
        else:
            bboxes = np.array(bboxes)
        results = [bboxes, lane_mask, area_mask] # , scores]
        self.bboxes = bboxes
        self.lane_mask = lane_mask
        self.area_mask = area_mask
        self.scores = scores
        if img_type:
            # 绘制查看效果
            self._get_img = img_type
            # lane_mask, area_mask, bboxes = results
            if img_type =='pil':
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            
            for bbox, score in zip(bboxes, scores):
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[2] + bbox[0]), int(bbox[3]+bbox[1])

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.putText(image, '%s:%.2f' % (self.classes[class_id-1], score), (x1, y1 - 5), 0,
                #            0.7, (0, 255, 0), 2)
            image[area_mask==1] = [0, 255, 0]
            image[lane_mask==1] = [255, 0, 0]
            if show:
                self.show(image)
            return results, image
        return results

    def load_context(self,context):
        self.context = context
        self.doc = read_squad_examples_context(input_data=context)
        return self.doc 

    def _nlp_qa_infer(self,data=None,context=None):
        if context is not None:
            self.contexts.append(context)
            doc = read_squad_examples_context(input_data=context)
        else:
            if self.doc is None:
                raise Exception(ecf.LOAD_CONTEXT_BEFORE_INFER())
            doc = self.doc
            self.contexts.append(self.context)


        if os.path.splitext(data)[1] == '.json':
            eval_examples = read_squad_examples_file(input_file=data)
            self.questions = []
            with open(data, "r") as f:
                input_data = json.load(f)["data"]
            print("inout",input_data)
            for idx, entry in enumerate(input_data):
                for paragraph in entry["paragraphs"]:
                    for qa in paragraph["qas"]:
                        qas_id = qa["id"]
                        question_text = qa["question"]
                        print("question_text",question_text)
                        self.questions.append(question_text)
        else:   
            self.questions.append(data)
            eval_examples = [squad_example(qas_id="1",question_text=data,doc_tokens=doc)]
        # eval_examples = read_squad_examples_file(input_file=predict_file)

        max_seq_length = 256
        doc_stride = 128
        max_query_length = 64
        batch_size = 1
        n_best_size = 20
        max_answer_length = 64

        vocab_file = os.path.join(os.path.dirname(__file__), 'tokenizer','qa_vocab.txt')
        # vocab_file = 'vocab.txt'
        tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

        input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(eval_examples, tokenizer, max_seq_length, doc_stride, max_query_length)

        n = len(input_ids)
        bs = batch_size
        all_results = []
        for idx in range(0, n):
            data = {"unique_ids_raw_output___9:0": np.array([1], dtype=np.int64),
                    "input_ids:0": input_ids[idx:idx+bs],
                    "input_mask:0": input_mask[idx:idx+bs],
                    "segment_ids:0": segment_ids[idx:idx+bs]}
            result = self.model.run(["unique_ids:0","unstack:0", "unstack:1"], data)
            in_batch = result[1].shape[0]
            start_logits = [float(x) for x in result[1][0].flat]
            end_logits = [float(x) for x in result[2][0].flat]
            for i in range(0, in_batch):
                unique_id = len(all_results)
                all_results.append(RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

        res, answer = write_predictions(eval_examples, extra_data, all_results,
                        n_best_size, max_answer_length,True)
        res = dict(res)["1"]
        self.answers.append(answer)
        return res
    
    def _baseml_infer(self,data=None, show=None, img_type=None):
        if isinstance(data, list):
            data = np.array(data)
        if data is not None and self.input_shape is not None: 
            model_input_shape = str(self.input_shape).replace(str(self.input_shape[0]), 'batch')
            assert type(self.demo_input) == type(data), f"Error Code: -309. The data type {type(data)} doesn't match the model input type {type(self.demo_input)}. Example input: {self.demo_input.tolist()}."
            assert self.input_shape[1:] == data.shape[1:], f"Error Code: -309. The data shape {data.shape} doesn't match the model input shape {model_input_shape}. Example input: {self.demo_input.tolist()}."

        if isinstance(self.model, List): # polynormial 任务会包括两个模型 [PolynomialFeatures(), LinearRegression()]
            res = self.model[0].transform(data)
            self.baseml_res = self.model[1].predict(res)
        else:
            self.baseml_res = self.model.predict(data)
        if img_type is not None:
            dummy_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),'none.jpg')
            img = cv2.imread(dummy_path)
            self._get_img = img_type
            warnings.warn("BaseML task will not return an image and it is meaningless to set the parameter 'img_type'. ")
            if show:
                self.show(img)
            return self.baseml_res, img
        return self.baseml_res

    def _style_custom_infer(self,data=None,show=None,img_type=None):

        def normalize(image, mean, std):
            img_data = image.astype(np.float32) / 255
            mean = np.array(mean, dtype=np.float32)
            std = np.array(std, dtype=np.float32)
            norm_img_data = (img_data - mean[:, None, None]) / std[:, None, None]
            return norm_img_data

        def inv_norm(img_data, mean, std):
            mean = np.array(mean, dtype=np.float32)
            std = np.array(std, dtype=np.float32)
            norm_img_data = img_data * std[:, None, None] + mean[:, None, None]
            return norm_img_data

        def preprocess(data):
            # c = cv2.imread(img)
            if isinstance(data, str):
                image = cv2.imread(data) 
            elif isinstance(data, np.ndarray):
                if len(data.shape) == 2: # grayscale
                    data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR) 
                image = data
            c = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            c = np.array(c)
            c = np.transpose(c,(2, 0, 1))
            c = normalize(c, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            c = np.expand_dims(c,0).astype("float32")
            return c

        def postprocess(out):
            out = inv_norm(out[0],[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            out = np.transpose(out,(1, 2, 0))
            out = np.clip(out,0,1)
            out = (out*255).astype("uint8")
            return out


        c = preprocess(data)
        s = preprocess(self.style_img)  

        out = self.model.run(None, {"content_img": c, "style_img": s, "alpha":np.array([1.0]).astype("float32")})[0]

        self.style_res = postprocess(out)
        # print(out.shape)
        if img_type:
            self._get_img = img_type
            if img_type =='cv2':
                # image = cv2.cvtcolor(image,cv2.BGR2RGB)
                self.style_res = cv2.cvtColor(self.style_res,cv2.COLOR_BGR2RGB)
            if show:
                self.show(self.style_res)
            return self.style_res,self.style_res
        return self.style_res

    def _style_infer(self,data=None,show=None,img_type=None):
        if isinstance(data, str):
            image = cv2.imread(data) 
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 2: # grayscale
                data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR) 
            image = data
        def preprocess(image):
            image = cv2.resize(image, (224,224))
            # Preprocess image
            x = np.array(image).astype('float32')
            x = np.transpose(x, [2, 0, 1])
            x = np.expand_dims(x, axis=0)
            return x 
        
        def postprocess(result,data):
            result = np.squeeze(result[0])
            result = np.clip(result, 0, 255)
            result = result.transpose(1,2,0).astype("uint8")
            # img = Image.fromarray(result)

            return result
        data = preprocess(image)

        ort_inputs = {'input1': data}
        # model = ort.InferenceSession(checkpoint, None)
        res = self.model.run(['output1'], ort_inputs)
        # print(self.custom_res[0].shape)
        self.style_res = postprocess(res,data)
        if img_type:
            self._get_img = img_type
            if img_type =='cv2':
                # image = cv2.cvtcolor(image,cv2.BGR2RGB)
                self.style_res = cv2.cvtColor(self.style_res,cv2.COLOR_BGR2RGB)
            if show:
                self.show(self.style_res)
            return self.style_res,self.style_res
        else:
            self._get_img = 'pil'
        return self.style_res

    def _cls_infer(self,data=None,show=None,img_type=None):
        def preprocess(input_data):
            # convert the input data into the float32 input
            input_data =cv2.resize(input_data,(224,224))
            input_data = np.transpose(input_data,(2,0,1))
            img_data = input_data.astype('float32')
            mean_vec = np.array([123.675,116.28,103.53])/255
            stddev_vec = np.array([58.395,57.12,57.375])/255
            norm_img_data = np.zeros(img_data.shape).astype('float32')
            for i in range(img_data.shape[0]):
                norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
            norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
            return norm_img_data
        
        if isinstance(data, str):
            image = cv2.imread(data) 
            data = preprocess(image)
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 2: # grayscale
                data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR) 
            image = data
            data = preprocess(data)

        ort_inputs = {'input': data}
        self.class_res = self.model.run(['output'], ort_inputs)[0]
        if img_type:
            # 绘制查看效果
            if img_type =='pil':
                # image = cv2.cvtcolor(image,cv2.BGR2RGB)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            if show:
                self.show(image)
            return self.class_res, image
        return self.class_res 

    def _custom_infer(self,data,preprocess=None, postprocess=None): 
        if preprocess is not None:
            data = preprocess(data)
        input_name = self.model.get_inputs()[0].name
        output_names = [o.name for o in self.model.get_outputs()]
        ort_inputs = {input_name: data}
        self.custom_res = self.model.run(output_names, ort_inputs)
        # print(self.custom_res[0].shape)
        if postprocess is not None:
            self.custom_res = postprocess(self.custom_res,data)
        return self.custom_res 

    def _basenn_infer(self,data=None,show=True, img_type=None):
        ort_session = self.model
        metamap = ort_session.get_modelmeta().custom_metadata_map
        try:
            input_size = ast.literal_eval(metamap['input_size'])
        except (ValueError, SyntaxError):
            input_size = tuple(map(int, metamap['input_size'].strip('()').split(',')))
        dataset_type = metamap['dataset_type']
        if dataset_type == 'img':
            if isinstance(data,str): # 文件路径
                data = cv2.imread(data)# .transpose(2,0,1)
                if input_size[1] == 1: # 灰度
                    data = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
                    data = cv2.resize(data, tuple(input_size[2:]))
                    data = np.expand_dims(data,2) # 增加channel维
                else:
                    data = cv2.resize(data,input_size[2:]) 

                data = np.expand_dims(data,0) # 增加batch维
                data = data.transpose(0,3,1,2) # （batch,channel,w,h）
            if isinstance(data,List):
                data = np.array(data)
            data  = data.astype(np.float32) 
            ort_inputs = {'input': data}
            # res = ort_session.run(['output'], ort_inputs)
        elif dataset_type == 'tab':
            if isinstance(data,List):
                data = np.array(data)
            data  = data.astype(np.float32) # tab iris
            ort_inputs = {'input': data}
            # res = ort_session.run(['output'], ort_inputs)
        elif dataset_type == 'npz':
            if isinstance(data, str): # 
                if os.path.splitext(data)[1] == '.npz': # action npz
                    data = np.load(data, allow_pickle=True)['data']
                    data  = data.astype(np.float32) # tab iris
                else: # tang zi
                    try:
                        word2idx = ast.literal_eval(metamap['word2idx'])
                    except (ValueError, SyntaxError):
                        word2idx = json.loads(metamap['word2idx'])
                    self.ix2word = {v:k for k, v in word2idx.items()}

                    data = [[word2idx[i] for i in data]]
            if isinstance(data,List):
                data = np.array(data)
            # data  = data.astype(np.float32) # tab iris
        ort_inputs = {'input': data}
        self.basenn_res = ort_session.run(['output'], ort_inputs)

        if img_type is not None:
            dummy_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),'none.jpg')
            img = cv2.imread(dummy_path)
            self._get_img = img_type
            warnings.warn("BaseNN task will not return an image and it is meaningless to set the parameter 'img_type'. ")
            if show:
                self.show(img)
            return self.basenn_res, img
        return self.basenn_res

    def _mmedu_infer(self,data=None,show=False,img_type=None,thr=None):
        if img_type is None:
            self.result = self.model.inference(data, show=show,get_img=img_type,score=thr)
            self.result = self.model.print_result(self.result)
            return self.result
        else:
            self.result,img = self.model.inference(data,get_img=img_type,score=thr)
            self.result = self.model.print_result(self.result)
            if show ==True:
                self.show(img)
            return self.result,img

    def _ocr_infer(self,data=None,show=False,img_type=None):

        result, elapse_list = self.model(data)
        if result is None:
            self.bboxs, self.classes, self.scores = [],[],[]
            data = cv2.imread(data) if isinstance(data,str) else data
            if show:
                self._get_img = 'cv2'
                self.show(data)
            return [],data
        self.bboxs, self.classes,    self.scores = list(zip(*result))
        result = list(zip(self.classes,self.bboxs))
        if img_type:
            from rapidocr_onnxruntime import VisRes
            path = os.path.join(self.asset_root, "font")
            font_file = os.path.join(path,"FZYTK.TTF")
            _migrate_legacy_file(font_file, "font")
            if not os.path.exists(font_file):
                os.makedirs(path, exist_ok=True)
                url = "https://www.openinnolab.org.cn/res/api/v1/file/creator/80ce50bf-68b6-4a3c-bc64-cc3a5dfd624b.ttf&name=ocr.ttf"
                downloader = Downloader(url, font_file,output_dir=path)
                downloader.start()
            vis = VisRes(text_score=0.2)
            res_img = vis(data, self.bboxs, self.classes, self.scores, font_path=font_file)
            if img_type == 'cv2':
                res_img = res_img
            elif img_type =='pil':
                res_img = cv2.cvtColor(res_img ,cv2.COLOR_BGR2RGB)
            if show:
                self.show(res_img)
            return result,res_img

        return result

    def _face_det_infer(self,data=None,show=False,get_img=None,**kwargs):

        if isinstance(data,str):
            image = cv2.imread(data)
        else:
            image = copy.copy(data)
        face_model = self.model
        use_legacy_params = any(key in kwargs for key in FACE_DET_LEGACY_PARAMS)
        is_cascade_model = hasattr(face_model, "detectMultiScale")

        self.bboxs = []
        self.scores = []
        self.classes = None

        if use_legacy_params or is_cascade_model:
            legacy_model = face_model if is_cascade_model else cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            scaleFactor = 1.1 if 'scaleFactor' not in kwargs else float(kwargs['scaleFactor'])
            minNeighbors = 5 if 'minNeighbors' not in kwargs else int(kwargs['minNeighbors'])
            minSize = (50,50) if 'minSize' not in kwargs else kwargs['minSize']
            maxSize = image.shape[:2] if 'maxSize' not in kwargs else kwargs['maxSize']
            faces = legacy_model.detectMultiScale(
                image,
                scaleFactor=scaleFactor,
                minNeighbors=minNeighbors,
                minSize=minSize,
                maxSize=maxSize,
            )
            for bbox in faces:
                ex_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                self.bboxs.append(ex_bbox)
        else:
            h, w = image.shape[:2]
            score_threshold = float(kwargs.get('score_threshold', kwargs.get('thr', 0.6)))
            nms_threshold = float(kwargs.get('nms_threshold', 0.3))
            top_k = int(kwargs.get('top_k', 5000))
            face_model.setInputSize((w, h))
            face_model.setScoreThreshold(score_threshold)
            face_model.setNMSThreshold(nms_threshold)
            face_model.setTopK(top_k)
            _, faces = face_model.detect(image)
            if faces is not None:
                for det in faces:
                    x, y, bw, bh = det[:4]
                    score = float(det[14]) if len(det) > 14 else None
                    ex_bbox = [float(x), float(y), float(x + bw), float(y + bh)]
                    self.bboxs.append(ex_bbox)
                    self.scores.append(score)

        h,w,c = image.shape
        sketch_scale = max(1,(min(h,w)/ 100))

        if get_img:
            if get_img =='cv2':
                for [a,b,c,d] in self.bboxs:
                    cv2.rectangle(image, (int(a),int(b)),(int(c),int(d)),(0,0,255),thickness=max(1,int(sketch_scale/2)))
            elif get_img =='pil':
                # image = cv2.cvtcolor(image,cv2.BGR2RGB)
                for [a,b,c,d] in self.bboxs:
                    cv2.rectangle(image, (int(a),int(b)),(int(c),int(d)),(0,0,255),thickness=max(1,int(sketch_scale/2)))
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            if show:
                self.show(image)
            return np.array(self.bboxs),image
        return np.array(self.bboxs)
    
    def _det_infer(self,data=None,show=False,get_img=None,threshold=0.5,target_class=None):
        def preprocess(input_data):
            # convert the input data into the float32 input
            img_data = input_data.astype('float32')

            #normalize
            mean_vec = np.array([0.485, 0.456, 0.406])
            stddev_vec = np.array([0.229, 0.224, 0.225])
            norm_img_data = np.zeros(img_data.shape).astype('float32')
            for i in range(img_data.shape[0]):
                norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

            #add batch channel
            norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
            return norm_img_data

        ### 读取模型
        model = self.model
        # session = ort.InferenceSession('rtmdet-acc0de.onnx', None)
        #注 实际根据不同的推理需求例如 TensorRT,CUDA,等需要更多设置，具体参考官网

        ### 准备数据
        # img_path = 'pose2.jpg'
        if isinstance(data,str):
            # image = plt.imread(data)
            # image = Image.open(data)
            image = cv2.imread(data)
        else:
            image = copy.copy(data)
            # image = Image.fromarray(data)
        # image = Image.open(img_path).resize((224,224))  # 这里设置的动态数据尺寸，不需要resize
        re_image =cv2.resize(image,(224,224))
        re_image_data = np.array(re_image).transpose(2, 0, 1)
        input_data = preprocess(re_image_data)

        ### 推理
        raw_result = model.run([], {'input': input_data})
        # print(raw_result[1])
        ### 后处理
        h_ratio =  image.shape[0] /224
        w_ratio =  image.shape[1] /224 
        self.bboxs = []
        self.scores = []
        self.classes = []
        for (idx,[a,b,c,d,e]) in enumerate(raw_result[0][0]):#,raw_result[1][0]:
            if target_class is not None:
                if isinstance(target_class,str):    
                    if coco_class[raw_result[1][0][idx]+1] != target_class:
                        continue
                elif isinstance(target_class,List):
                    if coco_class[raw_result[1][0][idx]+1] not in  target_class:
                        continue
            if e> threshold:                    
                bbox = [a*w_ratio,b*h_ratio,c*w_ratio,d*h_ratio]
                self.bboxs.append(bbox)
                self.scores.append(e)
                self.classes.append(coco_class[raw_result[1][0][idx]+1])
        if get_img:
            if get_img =='cv2':
                for i,[a,b,c,d] in enumerate(self.bboxs):
                    # cv2.rectangle(image, (int(a),int(b)),(int(c),int(d)),(0,0,255),2)
                    if self.task in ['det_coco','det_coco_l']:
                        # cv2.putText(image, str(self.classes[i]), (int(a),int(b)),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255), 2)
                        # (w,h), _ = cv2.getTextSize(str(self.classes[i]),cv2.FONT_HERSHEY_COMPLEX,1,2)
                        # print(w,h)
                        image = pil_draw(image, bbox1=(int(a), int(b), int(c), int(d)), label1=str(self.classes[i]))
                    # cv2.rectangle(image, (int(a),int(b)),(int(c),int(d)),(0,0,0),2)
                    else:
                        cv2.rectangle(image, (int(a),int(b)),(int(c),int(d)),(0,0,255),2)


            elif get_img =='pil':
                # image = cv2.cvtcolor(image,cv2.BGR2RGB)
                for i, [a,b,c,d] in enumerate(self.bboxs):
                    if self.task in ['det_coco','det_coco_l']:
                        image = pil_draw(image, bbox1=(int(a), int(b), int(c), int(d)), label1=str(self.classes[i]))
                    else:
                        cv2.rectangle(image, (int(a),int(b)),(int(c),int(d)),(0,0,255),2)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            if show:
                self.show(image)
            return np.array(self.bboxs),image
        return np.array(self.bboxs)
        
    def _pose_infer(self,data=None,show=False,get_img=None,bbox=None):
        model = self.model
        self.bbox=bbox
        h, w = model.get_inputs()[0].shape[2:]
        model_input_size = (w, h)
        if isinstance(data,str):
            img = cv2.imread(data)
        else:
            img = data
        # 前处理
        # start_time = time.time()
        self.data = data
        resized_img, center, scale = mmpose_preprocess(img, model_input_size,bbox)
        input_tensor = [resized_img.transpose(2, 0, 1)]
        input_name = model.get_inputs()[0].name
        output_names = [o.name for o in model.get_outputs()]
        # end_time = time.time()
        # print('前处理耗时：',end_time-start_time)
        # 模型推理
        # start_time = time.time()
        outputs = model.run(output_names, {input_name: input_tensor})
        # end_time = time.time()
        # print('推理耗时：',end_time-start_time)
        # 后处理
        # start_time = time.time()
        self.keypoints, self.scores = mmpose_postprocess(outputs, model_input_size, center, scale)
        # end_time = time.time()
        # print('后处理耗时：',end_time-start_time)
        # print('推理结果：')
        # print(keypoints)
        if get_img:
            # 绘制查看效果
            re = self._get_image()
            if show:
                self.show(re)
            return self.keypoints[0], re
        return self.keypoints[0]
    
    def _get_image(self):
        sketch = {}
        ratio = 1
        if self.task == 'pose_hand21':
            sketch = {
                'red':[[0,1],[1,2],[2,3],[3,4]],
                'orange':[[0,5],[5,6],[6,7],[7,8]],
                'yellow':[[0,9],[9,10],[10,11],[11,12]],
                'green':[[0,13],[13,14],[14,15],[15,16]],
                'blue':[[0,17],[17,18],[18,19],[19,20]]
            }
        elif self.task =='pose_body26':
            sketch = {
                'red':[[0,1],[1,2],[2,0],[2,4],[1,3],[0,17],[0,18]],
                'orange':[[18,6],[8,6],[10,8]],
                'yellow':[[18,19],[19,12],[19,11]],
                'green':[[12,14],[14,16],[16,23],[21,16],[25,16]],
                'blue':[[11,13],[13,15],[15,20],[15,22],[15,24]],
                'purple':[[18,5],[5,7],[7,9]],
            }
        elif self.task in['pose_body17','pose_body17_l']:
            sketch = {
                'red':[[0,1],[1,2],[2,0],[2,4],[1,3],[4,6],[3,5]],
                'orange':[[10,8],[8,6]],
                'yellow':[[5,6],[6,12],[12,11],[11,5]],
                'green':[[12,14],[14,16]],
                'blue':[[11,13],[13,15]],
                'purple':[[5,7],[9,7]],
            }
        elif self.task=='pose_wholebody133':
            sketch = {
                # body
                'red':[[0,1],[1,2],[2,0],[2,4],[1,3],[4,6],[3,5]],
                'orange':[[10,8],[8,6]],
                'yellow':[[5,6],[6,12],[12,11],[11,5]],
                'green':[[12,14],[14,16]],
                'blue':[[11,13],[13,15]],
                'purple':[[5,7],[9,7]],
            }
            ratio = 0.3
        elif self.task == 'pose_face106':
            ratio = 0.3
        if isinstance(self.data,str):
            img = cv2.imread((self.data))
        else:
            img = copy.copy(self.data)
        h,w,c = img.shape
        sketch_scale = max(1,(min(h,w)/ 100))

        point_scale = max(1,int(sketch_scale*ratio))

        # sketch 
        for color in sketch.keys():
            # print(color,sketch[color])
            for [fx,fy] in sketch[color]:
                # plt.plot([self.keypoints[0][fx][0],self.keypoints[0][fy][0]],[self.keypoints[0][fx][1],self.keypoints[0][fy][1]],color=color)
                cv2.line(img, (int(self.keypoints[0][fx][0]),int(self.keypoints[0][fx][1])),(int(self.keypoints[0][fy][0]),int(self.keypoints[0][fy][1])),color=self.color_dict[color],thickness=max(1,int(sketch_scale/2)))

        # keypoints
        for j in range(self.keypoints.shape[0]):
            for i in range(self.keypoints.shape[1]):
                # plt.scatter(self.keypoints[j][i][0],self.keypoints[j][i][1],c='b',s=10)
                x1,y1 = self.keypoints[j][i]
                if self.bbox is not None and self.erase:
                    sx1,sy1,sx2,sy2 = self.bbox # 
                    if sx1<x1<sx2 and sy1<y1<sy2 :
                        cv2.circle(img,(int(self.keypoints[j][i][0]),int(self.keypoints[j][i][1])),radius=max(1,int(point_scale/2)),color=[0,255,255],thickness=int(point_scale))
                else:
                    cv2.circle(img,(int(self.keypoints[j][i][0]),int(self.keypoints[j][i][1])),radius=max(1,int(point_scale/2)),color=[0,255,255],thickness=int(point_scale))

        if self._get_img == 'cv2':
            return img
        elif self._get_img.lower() == 'pil':
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def show(self,img): # check
        if isinstance(img,tuple):
            img = img[1]
        try:
            if self._get_img.lower() == 'cv2':
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif self._get_img.lower() == 'pil':
                img = img
            # plt.clf()
            plt.imshow(img, cmap='gray')
            plt.show()
        except:
            raise Exception(ecf.INFER_BEFORE_SHOW())
    
    def save(self,img,save_path): # check
        if self._get_img == 'cv2':
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if isinstance(img,tuple):
                img = img[1]
            if img.dtype != 'uint8':
                img = (img*255).astype('uint8')
            cv2.imwrite(save_path,img)
        elif self._get_img.lower() == 'pil':
            if isinstance(img,tuple):
                img = img[1]
            img = img
            if img.dtype != 'uint8':
                img = (img*255).astype('uint8')
            a = Image.fromarray(img)
            # 如果savepath以jpg结尾，则将rgbaa准华为rgb
            if save_path.split('.')[-1].lower() in ['jpg','jpeg']:
                a = a.convert('RGB')
            a.save(save_path)
            # plt.imshow(img)
            # plt.axis('off')
            # plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
            # plt.margins(0,0)
            # plt.savefig(save_path)
    
    def format_output(self,lang='zh',isprint=True, **kwargs): # check
        language = lang
        if self.task in ["pose_body17","pose_body26","pose_face106","pose_hand21","pose_wholebody133","pose_body17_l"]:
            formalize_keys = {
                "zh":["关键点坐标","分数"],
                "en":["keypoints","scores"],
                "ru":["ключевые точки","баллы"],
                "de":["Schlüsselpunkte","Partituren"],
                "fr":["points clés","partitions"]
            }
            formalize_result = {
                formalize_keys[language][0]:self.keypoints[0].tolist(),
                formalize_keys[language][1]:self.scores[0].tolist(),
            }
            # formalize_result = json.dumps(formalize_result, indent=4,sort_keys=False)
        elif self.task in['det_body','det_coco','det_hand','det_body_l','det_coco_l',]:
            formalize_keys = {
                "zh":["检测框","分数","类别"],
                "en":['bounding boxes',"scores","class"],
                "ru":["ограничивающие рамки","баллы","занятия"],
                "de":["Begrenzungsrahmen","Partituren","Klassen"],
                "fr":["cadres de délimitation","partitions","Des classes"]
            }
            if self.task == 'det_coco' and self.classes is not None:
                formalize_result = {
                    formalize_keys[language][0]:self.bboxs,
                    formalize_keys[language][1]:self.scores,
                    formalize_keys[language][2]:self.classes,
                }
            else:
                formalize_result = {
                    formalize_keys[language][0]:self.bboxs,
                    formalize_keys[language][1]:self.scores,
                }
        elif self.task in['det_face']:
            formalize_keys = {
                "zh":["检测框","分数","类别"],
                "en":['bounding boxes',"scores","class"],
                "ru":["ограничивающие рамки","баллы","занятия"],
                "de":["Begrenzungsrahmen","Partituren","Klassen"],
                "fr":["cadres de délimitation","partitions","Des classes"]
            }
            if self.classes is not None:
                formalize_result = {
                    formalize_keys[language][0]:self.bboxs,
                    formalize_keys[language][1]:self.scores,
                    formalize_keys[language][2]:self.classes,
                }
            elif len(self.scores) > 0:
                formalize_result = {
                    formalize_keys[language][0]:self.bboxs,
                    formalize_keys[language][1]:self.scores,
                }
            else:
                formalize_result = {
                    formalize_keys[language][0]:self.bboxs,
                }
        elif self.task in ['ocr']:
            formalize_keys = {
                "zh":["检测框","分数","文本"],
                "en":['bounding boxes',"scores","text"],
                "ru":["ограничивающие рамки","баллы","текст"],
                "de":["Begrenzungsrahmen","Partituren","Text"],
                "fr":["cadres de délimitation","partitions","texte"]
            }
            formalize_result = {
                formalize_keys[language][0]:list(self.bboxs),
                formalize_keys[language][1]:list(self.scores),
                formalize_keys[language][2]:list(self.classes),
            }
        elif self.task in ['mmedu']:
            formalize_result = self.result
        elif self.task in ['basenn']:
            formalize_keys = {
                "zh":["预测值","分数"],
                "en":["prediction","scores"],
                "ru":["прогноз","баллы"],
                "de":["Vorhersage","Partituren"],
                "fr":["prédiction","partitions"]
            }
            res_idx = self.basenn_res[0].argmax(axis=1)
            formalize_result = {}
            for i,idx in enumerate(res_idx):
                try:
                    pred = self.ix2word[idx]
                except:
                    pred = idx
                formalize_result[i] ={formalize_keys[lang][0]:pred,formalize_keys[lang][1]:self.basenn_res[0][i][idx]} 
        elif self.task in ['baseml']:
            formalize_keys = {
                "zh":["预测值","分数"],
                "en":["prediction","scores"],
                "ru":["прогноз","баллы"],
                "de":["Vorhersage","Partituren"],
                "fr":["prédiction","partitions"]
            }
            formalize_result = {
                formalize_keys[lang][0]:self.baseml_res
            }
        elif self.task in ['cls_imagenet']:
            formalize_keys = {
                "zh":["预测值","分数","预测类别"],
                "en":["prediction","scores","category"],
                "ru":["прогноз","баллы","Kategorie"],
                "de":["Vorhersage","Partituren","Kategorie"],
                "fr":["prédiction","partitions","catégorie"]
            }
            res_idx = np.argmax(self.class_res)
            formalize_result = {
                formalize_keys[language][0]:res_idx,
                formalize_keys[language][1]:self.class_res[0][res_idx],
                formalize_keys[language][2]:imagenet_class[res_idx],
            }
        elif self.task in ['gen_style_mosaic','gen_style_candy','gen_style_rain-princess','gen_style_udnie','gen_style_pointilism','gen_style_custom']:
            formalize_result = self.style_res
        elif self.task in ['gen_color']:
            formalize_result = self.color_res
        elif self.task in ['depth_anything']:
            formalize_result = self.depth_res
        elif self.task in ['nlp_qa']:
            formalize_keys = {
                "zh":["问题","回答","文本","分数","上下文"],
                "en":["question","answer","text","score","context"],
                "ru":["вопрос","ответ","текст","баллы","контекст"],
                "de":["Frage","Antwort","Text","Partituren","Kontext"],
                "fr":["question","réponse","texte","partitions","contexte"]
            }
            formalize_results = []
            for i, question in enumerate(self.questions):
                answer = self.answers[i]["1"]
                answer = [{formalize_keys[language][2]:i['text'],formalize_keys[language][3]:i['probability']}  for i in answer if i['probability'] >0.01]
                if kwargs.get('show_context',False):
                    formalize_result = {    
                        formalize_keys[language][4]:self.contexts[i],
                        formalize_keys[language][0]:question,
                        formalize_keys[language][1]:answer,
                    }
                else:
                    formalize_result = {    
                        formalize_keys[language][0]:question,
                        formalize_keys[language][1]:answer,
                    }
                formalize_results.append(formalize_result)
        elif self.task in ['drive_perception']:
            formalize_keys = {  
                "zh":["检测框","分数","车道线掩码","可行驶区域掩码"],
                "en":['bounding boxes',"scores","lane line mask","drivable area mask"],
                "ru":["ограничивающие рамки","баллы","маска линии полосы","маска области движения"],
                "de":["Begrenzungsrahmen","Partituren","Fahrspurmasken","Fahrbare Bereichsmaske"],
                "fr":["cadres de délimitation","partitions","masque de ligne de voie","masque de zone praticable"]
            }
            formalize_result = {
                formalize_keys[language][0]:self.bboxes.tolist(),
                formalize_keys[language][1]:np.squeeze(self.scores).tolist(),
                formalize_keys[language][2]:self.lane_mask,
                formalize_keys[language][3]:self.area_mask,
            }
        elif self.task in ['segment_anything']:
            formalize_keys = {
                "zh":["掩码","分数"],
                "en":["masks","scores"],
                "ru":["маски","баллы"],
                "de":["Maske","Partituren"],
                "fr":["masques","partitions"]
            }
            formalize_result = {
                formalize_keys[language][0]:self.masks,
                formalize_keys[language][1]:self.mask_scores.tolist(),
            }
        if isprint:
            try:
                pprint.pprint(formalize_result,sort_dicts=False)
            except:
                pprint.pprint(formalize_result)
        return formalize_result

from PIL import Image, ImageDraw, ImageFont
def pil_draw(img, bbox1, label1):
    # 1.读取图片
    im = Image.fromarray(img)
    font_path = os.path.join(os.path.join(os.path.dirname(__file__), 'font'),'FZYTK.TTF')
    # font_path = os.path.join(os.getcwd(), 'font/FZYTK.TTF')
    font = ImageFont.truetype(font=font_path, size=np.floor(1.5e-2 * np.shape(im)[1] + 15).astype('int32'))

    draw = ImageDraw.Draw(im)
    # 获取label长宽
    label_size1 = draw.textsize(label1, font)
    # label_size2 = draw.textsize(label2, font)

    # 设置label起点
    text_origin1 = np.array([bbox1[0], bbox1[1] - label_size1[1]])
    # text_origin2 = np.array([bbox2[0], bbox2[1] - label_size2[1]])

    # 绘制矩形框，加入label文本
    draw.rectangle([bbox1[0], bbox1[1], bbox1[2], bbox1[3]],outline='red',width=2)
    draw.rectangle([tuple(text_origin1), tuple(text_origin1 + label_size1)], fill='red')
    draw.text(text_origin1, str(label1), fill=(255, 255, 255), font=font)

    return np.asarray(im)
    

def bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    # get bbox center and scale
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale

def value2list(value: Any, valid_type: Union[Type, Tuple[Type, ...]],
               expand_dim: int) -> List[Any]:
    """If the type of ``value`` is ``valid_type``, convert the value to list
    and expand to ``expand_dim``.

    Args:
        value (Any): value.
        valid_type (Union[Type, Tuple[Type, ...]): valid type.
        expand_dim (int): expand dim.

    Returns:
        List[Any]: value.
    """
    if isinstance(value, valid_type):
        value = [value] * expand_dim
    return value


def check_type(name: str, value: Any,
               valid_type: Union[Type, Tuple[Type, ...]]) -> None:
    """Check whether the type of value is in ``valid_type``.

    Args:
        name (str): value name.
        value (Any): value.
        valid_type (Type, Tuple[Type, ...]): expected type.
    """
    if not isinstance(value, valid_type):
        raise TypeError(f'`{name}` should be {valid_type} '
                        f' but got {type(value)}')


def check_length(name: str, value: Any, valid_length: int) -> None:
    """If type of the ``value`` is list, check whether its length is equal with
    or greater than ``valid_length``.

    Args:
        name (str): value name.
        value (Any): value.
        valid_length (int): expected length.
    """
    if isinstance(value, list):
        if len(value) < valid_length:
            raise AssertionError(
                f'The length of {name} must equal with or '
                f'greater than {valid_length}, but got {len(value)}')


def check_type_and_length(name: str, value: Any,
                          valid_type: Union[Type, Tuple[Type, ...]],
                          valid_length: int) -> None:
    """Check whether the type of value is in ``valid_type``. If type of the
    ``value`` is list, check whether its length is equal with or greater than
    ``valid_length``.

    Args:
        value (Any): value.
        legal_type (Type, Tuple[Type, ...]): legal type.
        valid_length (int): expected length.

    Returns:
        List[Any]: value.
    """
    check_type(name, value, valid_type)
    check_length(name, value, valid_length)


def color_val_matplotlib(
    colors: Union[str, tuple, List[Union[str, tuple]]]
) -> Union[str, tuple, List[Union[str, tuple]]]:
    """Convert various input in RGB order to normalized RGB matplotlib color
    tuples,
    Args:
        colors (Union[str, tuple, List[Union[str, tuple]]]): Color inputs
    Returns:
        Union[str, tuple, List[Union[str, tuple]]]: A tuple of 3 normalized
        floats indicating RGB channels.
    """
    if isinstance(colors, str):
        return colors
    elif isinstance(colors, tuple):
        assert len(colors) == 3
        for channel in colors:
            assert 0 <= channel <= 255
        colors = [channel / 255 for channel in colors]
        return tuple(colors)
    elif isinstance(colors, list):
        colors = [
            color_val_matplotlib(color)  # type:ignore
            for color in colors
        ]
        return colors
    else:
        raise TypeError(f'Invalid type for color: {type(colors)}')


def color_str2rgb(color: str) -> tuple:
    """Convert Matplotlib str color to an RGB color which range is 0 to 255,
    silently dropping the alpha channel.

    Args:
        color (str): Matplotlib color.

    Returns:
        tuple: RGB color.
    """
    import matplotlib
    rgb_color: tuple = matplotlib.colors.to_rgb(color)
    rgb_color = tuple(int(c * 255) for c in rgb_color)
    return rgb_color


def convert_overlay_heatmap(feat_map: np.ndarray,
                            img: Optional[np.ndarray] = None,
                            alpha: float = 0.5) -> np.ndarray:
    """Convert feat_map to heatmap and overlay on image, if image is not None.

    Args:
        feat_map (np.ndarray): The feat_map to convert
            with of shape (H, W), where H is the image height and W is
            the image width.
        img (np.ndarray, optional): The origin image. The format
            should be RGB. Defaults to None.
        alpha (float): The transparency of featmap. Defaults to 0.5.

    Returns:
        np.ndarray: heatmap
    """
    assert feat_map.ndim == 2 or (feat_map.ndim == 3
                                  and feat_map.shape[0] in [1, 3])
    if feat_map.ndim == 3:
        feat_map = feat_map.transpose(1, 2, 0)

    norm_img = np.zeros(feat_map.shape)
    norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    if img is not None:
        heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    return heat_img

def wait_continue(figure, timeout: float = 0, continue_key: str = ' ') -> int:
    """Show the image and wait for the user's input.

    This implementation refers to
    https://github.com/matplotlib/matplotlib/blob/v3.5.x/lib/matplotlib/_blocking_input.py

    Args:
        timeout (float): If positive, continue after ``timeout`` seconds.
            Defaults to 0.
        continue_key (str): The key for users to continue. Defaults to
            the space key.

    Returns:
        int: If zero, means time out or the user pressed ``continue_key``,
            and if one, means the user closed the show figure.
    """  # noqa: E501
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import CloseEvent
    is_inline = 'inline' in plt.get_backend()
    if is_inline:
        # If use inline backend, interactive input and timeout is no use.
        return 0

    if figure.canvas.manager:  # type: ignore
        # Ensure that the figure is shown
        figure.show()  # type: ignore

    while True:

        # Connect the events to the handler function call.
        event = None

        def handler(ev):
            # Set external event variable
            nonlocal event
            # Qt backend may fire two events at the same time,
            # use a condition to avoid missing close event.
            event = ev if not isinstance(event, CloseEvent) else event
            figure.canvas.stop_event_loop()

        cids = [
            figure.canvas.mpl_connect(name, handler)  # type: ignore
            for name in ('key_press_event', 'close_event')
        ]

        try:
            figure.canvas.start_event_loop(timeout)  # type: ignore
        finally:  # Run even on exception like ctrl-c.
            # Disconnect the callbacks.
            for cid in cids:
                figure.canvas.mpl_disconnect(cid)  # type: ignore

        if isinstance(event, CloseEvent):
            return 1  # Quit for close.
        elif event is None or event.key == continue_key:
            return 0  # Quit for continue.

def img_from_canvas(canvas: 'FigureCanvasAgg') -> np.ndarray:
    """Get RGB image from ``FigureCanvasAgg``.

    Args:
        canvas (FigureCanvasAgg): The canvas to get image.

    Returns:
        np.ndarray: the output of image in RGB.
    """
    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    return rgb.astype('uint8')

def _fix_aspect_ratio(bbox_scale: np.ndarray,
                      aspect_ratio: float) -> np.ndarray:
    """Extend the scale to match the given aspect ratio.

    Args:
        scale (np.ndarray): The image scale (w, h) in shape (2, )
        aspect_ratio (float): The ratio of ``w/h``

    Returns:
        np.ndarray: The reshaped image scale in (2, )
    """
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c

def get_warp_matrix(center: np.ndarray,
                    scale: np.ndarray,
                    rot: float,
                    output_size: Tuple[int, int],
                    shift: Tuple[float, float] = (0., 0.),
                    inv: bool = False) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # compute transformation matrix
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    # get four corners of the src rectangle in the original image
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # get four corners of the dst rectangle in the input image
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat

def top_down_affine(input_size: dict, bbox_scale: dict, bbox_center: dict,
                    img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bbox image as the model input by affine transform.

    Args:
        input_size (dict): The input size of the model.
        bbox_scale (dict): The bbox scale of the img.
        bbox_center (dict): The bbox center of the img.
        img (np.ndarray): The original image.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: img after affine transform.
        - np.ndarray[float32]: bbox scale after affine transform.
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # reshape bbox to fixed aspect ratio
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    # get the affine matrix
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    # do affine transform
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale

def mmpose_preprocess(
    img: np.ndarray, input_size: Tuple[int, int] = (192, 256),bbox=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Do preprocessing for RTMPose model inference.

    Args:
        img (np.ndarray): Input image in shape.
        input_size (tuple): Input image size in shape (w, h).

    Returns:
        tuple:
        - resized_img (np.ndarray): Preprocessed image.
        - center (np.ndarray): Center of image.
        - scale (np.ndarray): Scale of image.
    """
    # get shape of image
    img_shape = img.shape[:2]
    if bbox is None:
        bbox = np.array([0, 0, img_shape[1], img_shape[0]])

    # get center and scale
    center, scale = bbox_xyxy2cs(bbox, padding=1.25)

    # do affine transformation
    resized_img, scale = top_down_affine(input_size, scale, center, img)

    # normalize image
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    resized_img = (resized_img - mean) / std

    return resized_img, center, scale

def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals

def mmpose_decode(simcc_x: np.ndarray, simcc_y: np.ndarray,
           simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
    """Modulate simcc distribution with Gaussian.

    Args:
        simcc_x (np.ndarray[K, Wx]): model predicted simcc in x.
        simcc_y (np.ndarray[K, Wy]): model predicted simcc in y.
        simcc_split_ratio (int): The split ratio of simcc.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: keypoints in shape (K, 2) or (n, K, 2)
        - np.ndarray[float32]: scores in shape (K,) or (n, K)
    """
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores

def mmpose_postprocess(outputs: List[np.ndarray],
                model_input_size: Tuple[int, int],
                center: Tuple[int, int],
                scale: Tuple[int, int],
                simcc_split_ratio: float = 2.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Postprocess for RTMPose model output.

    Args:
        outputs (np.ndarray): Output of RTMPose model.
        model_input_size (tuple): RTMPose model Input image size.
        center (tuple): Center of bbox in shape (x, y).
        scale (tuple): Scale of bbox in shape (w, h).
        simcc_split_ratio (float): Split ratio of simcc.

    Returns:
        tuple:
        - keypoints (np.ndarray): Rescaled keypoints.
        - scores (np.ndarray): Model predict scores.
    """
    # use simcc to decode
    simcc_x, simcc_y = outputs
    keypoints, scores = mmpose_decode(simcc_x, simcc_y, simcc_split_ratio)

    # rescale keypoints
    keypoints = keypoints / model_input_size * scale + center - scale / 2

    return keypoints, scores
