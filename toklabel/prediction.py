from __future__ import annotations
import abc
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Sequence, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

class Annotation(ABC):
    """
    任意标注返回 Label Studio result 统一接口

    属性：
    label_group 对应 label studio中的from_name
    label_item 对应 label studio中的to_name
    label_type 对应 label种类
    """
    label_group: str
    label_item:   str
    label_type:     str            # 人类可读标签

    @abstractmethod
    def to_ls(self) -> Dict:
        pass

@dataclass
class DataContext:
    '''
    数据上下文，目前包括shot，frame_time等
    '''
    shot: int | None = None
    frame_time: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass
class TimeseriesSpan(Annotation):
    start: float
    end:   float
    label_choice: str
    label_group: str = "ts_label"
    label_target: str = "ts"
    label_type: str = field(init=False, default="timeserieslabels")

    def to_ls(self) -> Dict:
        end = self.end if self.end is not None else self.start
        return {
            "from_name": self.label_group,
            "to_name":   self.label_target,
            "type":      self.label_type,
            "value": {
                "start": self.start,
                "end":   end,
                "timeserieslabels": [self.label_choice],
            }
        }

@dataclass
class Number(Annotation):
    value: float|int
    label_group: str = "num"
    label_target: str = "image"
    label_type: str = field(init=False, default="number")

    def to_ls(self) -> Dict:
        return {
            "from_name": self.label_group,
            "to_name":   self.label_target,
            "type":      self.label_type,
            "value": {
                "number":   self.value,
            }
        }


@dataclass
class BrushMask(Annotation):
    rle_mask: List[int]
    label_choice: str
    label_group: str = "tag"
    label_target: str = "image"
    image_height: int = 800
    image_width: int = 1280
    label_type: str = field(init=False, default="brushlabels")
    

    def to_ls(self) -> Dict:
        return {
            "from_name": self.label_group,
            "to_name":   self.label_target,
            "type":      self.label_type,
            'readonly': False,
            "image_rotation": 0,
            'original_width':  self.image_width,
            'original_height': self.image_height,
            "value": {
                "rle": self.rle_mask,
                "format": "rle",
                'brushlabels': [self.label_choice],
            }
        }

@dataclass
class Polygon(Annotation):
    poly_points: List[Tuple]
    label_choice: str
    label_group: str = "polytag"
    label_target: str = "image"
    image_height: int = 800
    image_width: int = 1280
    label_type: str = field(init=False, default="polygonlabels")
    

    def to_ls(self) -> Dict:
        return {
            "from_name": self.label_group,
            "to_name":   self.label_target,
            "type":      self.label_type,
            'readonly': False,
            "image_rotation": 0,
            'original_width':  self.image_width,
            'original_height': self.image_height,
            "value": {
                "points": self.poly_points,
                'polygonlabels': [self.label_choice],
            }
        }

@dataclass
class KeyPoint(Annotation):
    key_points: Tuple
    label_choice: str
    label_group: str = "keytag"
    label_target: str = "image"
    image_height: int = 800
    image_width: int = 1280
    point_width: float = 0.25
    label_type: str = field(init=False, default="keypointlabels")
    
    def to_ls(self) -> Dict:
        return {
            "from_name": self.label_group,
            "to_name":   self.label_target,
            "type":      self.label_type,
            'readonly': False,
            "image_rotation": 0,
            'original_width':  self.image_width,
            'original_height': self.image_height,
            "value": {
                "x": round(self.key_points[0] / self.image_width * 100, 2),
                "y": round(self.key_points[1] / self.image_height * 100, 2),
                "width": self.point_width,
                'keypointlabels': [self.label_choice],
            }
        }

class BasePredictor(ABC):
    """
    针对“一条任务数据”返回一组 Annotation。
    """
    @abstractmethod
    def predict(
        self, 
        task_data, 
        data_context:DataContext|None = None,
        **kwargs
    ) -> List[Annotation]:
        ...


def to_labelstudio_form(
    annotations: List[Annotation],
    model_version: str = "script"
) -> List[Dict]:
    """
    把任意 Annotation 子类列表转成 LS `predictions` 字段。
    """
    results = [ann.to_ls() for ann in annotations]
    return [{
        "model_version": model_version,
        "result": results,
    }] if results else []


def start_end_time_1D(predict_result, thredhold:float, postive :bool = True):
    """
    将预测值的1D序列和阈值转化为（起始时间、结束时间）
    """
    # 将数据都转化为numpy array
    predict_result = data_process(predict_result)
    thredhold = data_process(thredhold)
    if postive:
        mask  = predict_result > thredhold
    else:
        mask  = predict_result < thredhold
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.insert(starts,0,0)
    if mask[-1]:
        ends = np.append(ends, len(predict_result) - 1)
    return list(zip(starts, ends))


def start_end_time(predict_result, threshold, postive :bool = True, all_dim :bool = True, time_axis: int = -1):
    """
    将 2D 或更高维 数据在时间轴上进行阈值检查，
    并返回对每条“曲线”或“序列”找到的 (start, end) 区间列表。

    参数:
    -----
    predict_result: 2D array-like
        形如 (N, T) 或 (T, N) 或更多排列（只要有2维度可以处理）；
    threshold: 2D array-like / 1D / float
        和 predict_result 做比较，也可只有1个值(广播)，也可与predict_result同shape。
    dt: float
        时间分辨率(这里暂未用到；保留以便后续换算真实时间)。
    positive:bool
        为True则统计大于阈值的区间，为False则统计小于阈值的区间。
    all_dim: bool
        True表示所有维度必须同时满足大于阈值。False表示对每个通道逐一判断。默认为True
    time_axis: int
        哪个轴是时间轴。默认 -1 表示最后一个维度。

    返回:
    -----
    all_intervals: list of list of (start, end)
        结构示例: [ [ (s1, e1), (s2, e2), ...],  # 第一个通道/行的区间
                    [ (s1, e1), (s2, e2), ...],  # 第二个通道/行
                    ...
                  ]
    """
    predict_result = data_process(predict_result)
    threshold = data_process(threshold)

    if hasattr(threshold, "shape") and threshold.ndim == predict_result.ndim:
        threshold = np.moveaxis(threshold, time_axis, -1)
    elif hasattr(threshold, "shape"):
        while threshold.ndim < predict_result.ndim:
            threshold = np.expand_dims(threshold, axis=-1)
    threshold_moved = threshold

    # 移动时间轴至最后并广播阈值
    predict_result_moved = np.moveaxis(predict_result, time_axis, -1)
    # 确保阈值广播到与predict_result_moved相同的形状
    predict_result_moved, threshold_moved = np.broadcast_arrays(predict_result_moved, threshold)

    if all_dim: #逻辑与，所有通道都应大于阈值
        if predict_result_moved.ndim > 1:
        # 合并除时间轴外的所有维度
            if postive:
                mask = np.all(predict_result_moved > threshold_moved, axis=tuple(range(predict_result_moved.ndim-1)))
            else:
                mask = np.all(predict_result_moved < threshold_moved, axis=tuple(range(predict_result_moved.ndim-1)))
        else:
            mask = predict_result_moved > threshold_moved
        # 全局单一掩码，直接处理
        intervals = start_end_time_1D(mask, 0.5, postive=True)
        return [intervals]
    else: #对每个通道逐一判断
    # 重塑为二维数组 (num_sequences, T)
        orig_shape = predict_result_moved.shape
        num_sequences = np.prod(orig_shape[:-1])
        T = orig_shape[-1]
        predict_2d = predict_result_moved.reshape(num_sequences, T)
        threshold_2d = threshold_moved.reshape(num_sequences, T)
        # 处理每个序列
        all_intervals = [
            start_end_time_1D(predict_2d[i], threshold_2d[i], postive)
            for i in range(num_sequences)
        ]
        # 恢复原始形状结构
        all_intervals = np.array(all_intervals, dtype=object).reshape(orig_shape[:-1]).tolist()
        return all_intervals
  
    
def data_process(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return data

class _LegacyPrediction:
    """
    以使用新方法实现，该数据类已弃用
    统一的预测结果数据结构，每个对象代表对一个“区间”或“片段”的预测。
    目前仅支持时序标签中，未来可能扩展更多字段。
    """
    def __init__(self, label_group: str, label: str, start: float, end: Optional[float] = None, score: Optional[float] = None):
        '''
        label与label_group要和项目的前端配置匹配
        '''
        self.label_group = label_group
        self.label = label   # 例如 "effective", "invalid", ...
        self.start = start
        self.end = end #end参数可选，对于仅预测转折点的标注，可以为None
        self.score = score   # 置信度可选,目前无作用

    def __repr__(self):
        return f"<ModelPrediction label={self.label}, start={self.start}, end={self.end}, score={self.score}>"

class _LegacyBasePredictor(abc.ABC):
    '''
    抽象类，用于预标注
    '''
    @abc.abstractmethod
    def predict(self, task_data: pd.DataFrame) -> List[_LegacyPrediction]:
        """
        输入一条任务数据, 输出一个(或多个)预测结果。
        """
        pass


def _convert_to_labelstudio_form(pred_results: List[_LegacyPrediction], model_version :str = "script_v1") -> List[dict]:
    """
    将ModelPrediction转化为LabelStudio标准的格式

    参数:
    ----
    pred_results: List[ModelPrediction]
        模型的预测结果，以ModelPrediction形式存储每一个预测，列表存储多个标签组的预测
    model_version: str
        预测程序的名称，默认为“script_v1”  
    返回：
    ----
    LS_predictions：List[dict]
    Label Studio标准格式的predictions
    """
        # 转成 LS 的 result
    LS_result_list = []
    for p in pred_results:
        if p.end is None: #标注的是时间点
            p.end = p.start
        LS_result_list.append({
            "from_name": p.label_group,
            "to_name": "ts",
            "type": "timeserieslabels",
            "value": {
                "start": p.start,
                "end": p.end,
                "timeserieslabels": [p.label]
            }
        })
        
    LS_predictions = []
    if LS_result_list:
        LS_predictions.append({
            "model_version": model_version,
            #"score": None,
            "result": LS_result_list
        })
        
        # 最终拼到 tasks
    return LS_predictions   


if __name__ == "__main__":
    input_data = np.random.randn(3,100)
    for i in range(3):
        print(input_data[i,:])
        print('----------------------------------')
    thredhold = [0.1,0.1,0.1]
    result = start_end_time(input_data, thredhold, all_dim=True, postive=True)
    print(result)