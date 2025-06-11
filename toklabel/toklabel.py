import label_studio_sdk
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import os
import json
import time
from . import prediction, utils, config ,projectbuilder, maskutils
from .projectbuilder import ProjectBuilder
from typing import Literal, List, Optional, Dict, Any
import re
from collections import defaultdict
from ._label_config_utils import _build_label, _ensure_auxiliary_labels, _normalize_groups, _pretty_xml, CHANNEL_COLORS, LABEL_COLORS
from .prediction import Annotation, BasePredictor, to_labelstudio_form

def connect_Label_Studio(API_key :str = config.LABEL_STUDIO_API_KEY, label_studio_URL :str = config.LABEL_STUDIO_URL):
    '''
    链接Label Studio客户端

    参数:
    ----
    API_key: str
        用于连接Label Studio客户端，用户身份的证明。可以在Label Studio的Account & Settings界面找到。
    label_studio_URL: str
        Label Studio的连接，默认为 http://dap0.lan:30400                      
    返回:
    ----
    ls_client: Label Studio客户端，涉及Label StudioAPI的操作均需要它
    '''
    ls_client = label_studio_sdk.LabelStudio(base_url=label_studio_URL, api_key=API_key)
    return ls_client

#------ 创建配置界面部分 -------
def create_timeseries_label_config(name_table_columns, label_groups):
    """
    创建XML配置，用于时间序列标签和数据通道的显示
    
    参数:
    ----
    name_table_columns: dict
        格式例如：
        {
            "some_name": ("table_name", ["col1", "col2", ...]),
            "other_name": ("other_table", ["colA", "colB", ...])
        }
    label_groups: list of dict|list of tuple
        每个元素若为dict，至少需要包括"name" "choices"两个key，例如：
        [{"name":"LabelGroup1", "choices":["Choice1", "Choice2"]},...]
        每个元素若为为tuple，则采取 (label_name, choices) 形式，例如：
        [("LabelGroup1", ["Choice1", "Choice2"]),...]

    返回:
    ----
    xml_config: str
        格式化后的XML配置字符串。
    """
    groups = _normalize_groups(label_groups, data_type='ts')
    # 创建根节点
    root = ET.Element("View")
    
    # 生成TimeSeriesLabels部分，使用内置颜色表为不同标签组分配背景颜色
    for g in groups:
        _build_label(root, g, LABEL_COLORS, data_type='ts')
    
    # 创建TimeSeries部分
    TimeSeries = ET.SubElement(root, "TimeSeries",
                            attrib={'name': 'ts', 
                                    'valueType': 'url', 
                                    'value': '$csv', 
                                    'sep': ',', 
                                    'timeColumn': 'time', 
                                    'timeDisplayFormat': ',.3', 
                                    'fixedScale': 'true',
                                    'overviewWidth':"50%"})
    
    # 针对每个 name 对应的通道组，为所有通道设置图例、显示格式和描边颜色
    for i, (name, channels) in enumerate(name_table_columns.items()):
        # 为当前通道组分配一种颜色（保证同组内的所有通道颜色一致）
        channel_color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
        for channel in channels[1]:
            if name =='view_data':
                channel_name = channel
            else:
                channel_name = f'{name}_{channel}'
            disp_format = ",.3"
            attrib = {
                "column": channel_name, 
                "displayFormat": disp_format, 
                "legend": channel_name, 
                "strokeColor": channel_color
            }
            ET.SubElement(TimeSeries, "Channel", attrib=attrib)
    
    # 格式化XML输出
    xml_config = _pretty_xml(root)
    print("XML配置:")
    print(xml_config)
    return xml_config


def create_image_label_config(label_groups,
                              using_SAM_support: bool = False) -> str:
    """
    创建XML配置，支持图像数据的显示和多种图像标签，并支持创建适用于SAM的图像标注UI

    参数：
    label_groups: list of dict|list of tuple
        每个元素若为dict，至少需要包括"name" "choices"两个key，例如：
        [{"name":"LabelGroup1", "choices":["Choice1", "Choice2"]},...]
        每个元素若为为tuple，则采取 (label_name, choices) 形式，例如：
        [("LabelGroup1", ["Choice1", "Choice2"]),...]
 
    using_SAM_support: bool 
        True 时使用三栏 UI；False 时使用标准 UI
    """
    groups = _normalize_groups(label_groups)   # 兼容化

    # ============ UI 外壳 =============
    if using_SAM_support:
        STYLE_CSS = """
            .main {font-family: Arial, sans-serif; background:#f5f5f5; margin:0; padding:20px;}
            .container{display:flex;justify-content:space-between;margin-bottom:20px;}
            .column{flex:1;padding:10px;background:#fff;border-radius:5px;box-shadow:0 2px 5px rgba(0,0,0,.1);text-align:center;}
            .column .title{margin:0;color:#333;}
            .column .label{margin-top:10px;padding:10px;background:#f9f9f9;border-radius:3px;}
            .image-container{width:100%;height:300px;background:#ddd;border-radius:5px;}
        """.strip()
        groups = _ensure_auxiliary_labels(groups)
        root = ET.Element("View")
        ET.SubElement(root, "Style").text = STYLE_CSS
        main  = ET.SubElement(root, "View", {"className": "main"})
        cols  = ET.SubElement(main, "View", {"className": "container"})

        # ---- 根据 label type 自动分配到三列 ----------
        col_map = {                         
            "BrushLabels":     "Choose Label",
            "KeyPointLabels":  "Use Keypoint",
            "RectangleLabels": "Use Rectangle"
        }
        col_nodes = {}                      # 缓存列节点，防重复创建

        def _get_col(title):
            if title not in col_nodes:
                box = ET.SubElement(cols, "View", {"className": "column"})
                ET.SubElement(box, "View", {"className": "title"}).text = title
                col_nodes[title] = ET.SubElement(box, "View",
                                                 {"className": "label"})
            return col_nodes[title]

        for g in groups:
            title = col_map.get(g["type"], "Other")
            _build_label(_get_col(title), g, LABEL_COLORS)

        # ---- 图片容器 ----
        img_wrap = ET.SubElement(main, "View", {"className": "image-container"})
        ET.SubElement(img_wrap, "Image", {
            "name": "image", "value": "$image",
            "zoom": "true", "zoomControl": "true"
        })

    # ============ 标准 Label Studio UI =============
    else:
        root = ET.Element("View")
        for g in groups:
            _build_label(root, g, LABEL_COLORS)
        ET.SubElement(root, "Image", {
            "name": "image", "value": "$image",
            "zoom": "true", "zoomControl": "true"
        })
    # 格式化XML输出
    xml_config = _pretty_xml(root)
    print("XML配置:")
    print(xml_config)
    return xml_config    

    
def create_project(ls, 
                   project_name :str,
                   data_type :str, 
                   description :str, 
                   xml_config=None, 
                   name_table_columns=None, 
                   label_groups=None,
                   using_SAM_support :bool=True):
    """
    根据数据类型提供xml_config或者提供name_table_column和label_table_column以创建一个新项目
    
    参数:
    ----
    ls: object
        连接Label Studio客户端后返回的变量
    project_name: str
        新项目的名称
    data_type: str
        数据类型，支持"Timeseries"和"Image"    
    description: str
        新项目的描述
    xml_config: str
        xml格式的str，用于配置前端界面。
    name_table_columns: dict
        格式例如：
        {
            "some_name": ("table_name", ["col1", "col2", ...]),
            "other_name": ("other_table", ["colA", "colB", ...])
        }
    label_groups: list of dict|list of tuple
        每个元素若为dict，至少需要包括"name" "choices"两个key，例如：
        [{"name":"LabelGroup1", "choices":["Choice1", "Choice2"]},...]
        每个元素若为为tuple，则采取 (label_name, choices) 形式，例如：
        [("LabelGroup1", ["Choice1", "Choice2"]),...]
    using_SAM_support: bool
        是否使用SAM辅助，默认为True，仅对图像数据有效            
    返回:
    ----
    proj: dict
        新项目的元数据，包括项目的id
    """
    # 检查：如果 xml_config 为 None，则必须同时提供 name_table_columns 和 label_table_columns
    if xml_config is None and name_table_columns is None and label_groups is None :
        raise ValueError("至少应提供 xml_config, name_table_columns 和 label_table_columns中的一种")
    
    # 1. 创建项目
    if xml_config is None:
        if data_type.lower() == 'timeseries':
            xml_config = create_timeseries_label_config(name_table_columns, label_groups)
        elif data_type.lower() == 'image':
            xml_config = create_image_label_config(label_groups, using_SAM_support=using_SAM_support)
        else:
            raise ValueError(f'Invalid data type:{data_type}, only timeseries and image are supported')    
    proj = ls.projects.create(
        title=project_name,
        description=description,
        label_config=xml_config,
        show_skip_button=True,
        show_instruction=True,
        show_annotation_history=True
    )
    return proj

#------ 导出并解析Label Studio标注结果部分 -------
def export_annotation(ls, project_id :int, title=None, json_min=True, exclude_skipped=True, only_with_annotation=True,include_annotation_history=False, keep_snapshot=False):
    """
    以json或者json_min格式导出标注数据
    
    参数:
    ----
    ls: object
        连接Label Studio客户端后返回的变量
    project_id: int
        项目编号
    title: str
        可选。快照名称    
    json_min: bool
        json_min=True则导出json_min,否则导出json格式。默认为True
    exclude_skipped: bool
        是否包括在Label Studio中被跳过的任务。True则包括跳过的任务。默认为True。    
    only_with_annotation: bool
        only_with_annotation=True则只导出有标注的数据,否则也导出无标注的数据。默认为True
    include_annotation_history: bool
        include_annotation_history=True则导出有历史标注,否则只导出当前标注。默认为False
    keep_snapshot: bool
        是否保存快照，默认为False。如果保存快照，则可以使用快照再次导出相同的标注数据，但快照本身占据存储空间                         
    返回:
    ----
    export_json: dict
        json或者json-min格式的标注数据
    """
    import time
    if title is None:
        date = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        title = 'snapshot_'+date
    task_filter = {"only_with_annotations":only_with_annotation}
    annotation_filter = {"usual":True}
    if exclude_skipped:
        task_filter['skipped'] = "exclude"
        annotation_filter['skipped'] = False 
    else:
        annotation_filter['skipped'] = True 
    snapshot = ls.projects.exports.create(project_id=project_id, title=title,
                                          task_filter_options= task_filter,
                                          annotation_filter_options=annotation_filter,
                                          serialization_options={"include_annotation_history": include_annotation_history})
    if json_min:
        res = ls.projects.exports.convert(project_id, snapshot.id, export_type="JSON_MIN", download_resources=True)
        annotations = ls.projects.exports.download(project_id, snapshot.id, export_type='JSON_MIN')
    else: 
        annotations = ls.projects.exports.download(project_id, snapshot.id)      
    annotation = b"".join(annotations)
    try:
        export_json = json.loads(annotation.decode())
    except json.JSONDecodeError as e:
        print("解析 JSON 数据时出错：", e)
    # 删除快照    
    if not keep_snapshot:
        ls.projects.exports.delete(project_id, snapshot.id)
    return export_json

def _get_shot_time(data_dict:Dict):
        '''
        提取炮号和帧时间
        '''
        shot = data_dict.get('shot', None)
        frame_time = data_dict.get('frame_time', None)
        if not shot or not frame_time:
            try:
                pattern = r"/(\d+)/(\d+)\.jpg"
                match = re.search(pattern, data_dict['image'])
                if not shot:
                    shot = int(match.group(1))
                if not frame_time:
                    frame_time = float(match.group(2)) / 1e5
            except:
               print('未读取到炮号和帧时间')
        return shot, frame_time

def parse_brush(res: dict, to_real_pixel:bool=True, shape:tuple=(800, 1280)):
    return {
        "geom_type": "mask",
        "geom_data": {"rle": res["rle"]},
        "label": res["brushlabels"][0]
    }

def parse_rectangle(res: dict,to_real_pixel:bool=True, shape:tuple=(800, 1280)):
    [x,y] = maskutils.percent_to_pixel((res['x'],res['y']),shape) if to_real_pixel else res['x'], res['y']
    [w,h] = maskutils.percent_to_pixel((res['weight'],res['height']),shape) if to_real_pixel else res['width'], res['height']
    return {
        "geom_type": "bbox",
        "geom_data": {
            "x": x, "y": y,
            "width": w, "height": h,
            "rotation": res.get("rotation", 0)
        },
        "label": res["rectanglelabels"][0]
    }

def parse_keypoint(res: dict, to_real_pixel:bool=True, shape:tuple=(800, 1280)):
    h, w = shape
    if to_real_pixel:
        x = int(round(res["x"] / 100 * w))
        y = int(round(res["y"] / 100 * h))
    else:    
        x = res['x']
        y = res['y']
    return {
        "geom_type": "point",
        "geom_data": {"x": x, "y": y},
        "label": res["keypointlabels"][0]
    }

def parse_polygon(res: dict, to_real_pixel:bool=True, shape:tuple=(800, 1280)):
    points_list = maskutils.percent_to_pixel(res['points'], shape) if to_real_pixel else res['points']
    return {
        "geom_type":"polygon",
        "geom_data":{"points": points_list},
        "label": res["polygonlabels"][0]
    }

def parse_number(res: dict, to_real_pixel:bool=True, shape:tuple=(800, 1280)):
    return{
        'geom_type':"number",
        'geom_data':{'number':res['number']}
    }

GEOMETRY_PARSERS = {
    "brushlabels":     parse_brush,
    "rectanglelabels": parse_rectangle,
    "keypointlabels":  parse_keypoint,
    "polygonlabels":   parse_polygon,
    "number":          parse_number,
}

def _extract_geometry(result: dict, to_real_pixel:bool=True, shape:tuple=(800,1280)):
    """
    根据 result['type'] (full json) 或关键字段(json_min) 挑选解析器
    """
    ltype = result.get("type") # full json 专有
    if ltype == 'textarea':
        return {}              
    if not ltype:
        # json_min 没有 type；根据字段猜
        if "text" in result:
            return {}
        elif "rle" in result:
            ltype = "brushlabels"
        elif "rectanglelabels" in result:
            ltype = "rectanglelabels"
        elif "points" in result:
            ltype = "polygonlabels"
        elif "keypointlabels" in result:
            ltype = "keypointlabels"
        else:
            raise ValueError(f"Un-recognized result keys: {result.keys()}")

    parser = GEOMETRY_PARSERS.get(ltype)
    if parser is None:
        raise NotImplementedError(f"No parser registered for {ltype}")

    return parser(result, to_real_pixel, shape)

def timeseries_json_convertor(annotation_json, 
                              json_min :bool = True, 
                              shot_unique = False, 
                              only_one_choice = False, 
                              multi_label_group :bool = False, 
                              label_group_name :str = None, 
                              include_no_annotation :bool = False,
                              taking_skipped_as_label:bool = False):
    """
    将Label Studio导出的时序数据的标注进一步化简，用于将数据保存至pgsql
    该函数只适用于label studio社区版，企业版的格式可能会有一定变化
    
    参数:
    ----
    annotation_json: dict
        export_annotation返回结果，或者其他以字典格式存储的json_min或json的标注数据 
    json_min : bool
        json_min为True则annotation_json为json_min语法，否则annotation_json为完整的json语法
    shot_unique: bool
        是否只允许一炮有一组标注结果，若multi_label_group为True或一个标注有多个时间段，仍可能返回同属一个标注者的一组多个标注结果
    only_one_choice: bool
        每个特征是否都只有一个选项，如果为True，则输出结果不包含label或{label_group_name}这个key，默认为False    
    multi_label_group : bool
        是否存在多组标签
    label_group_name: str
        指定单组标签的名称，默认为“label”。仅在multi_label_group=False时有效
    include_no_annotation: bool
        是否导出无标注的任务，仅在json_min=False时有效。若include_no_annotation=True，则包含无标注任务的炮号
    taking_skipped_as_label: bool
        是否将skipped也作为一种label，仅在json_min=False时有效。若taking_skipped_as_label=True，则将被跳过作为标签加入解析结果。    
    返回:
    ----
    label_list: List[dict]
        简化的标注结果，用于插入Postgresql数据库的形式
    """

    def get_shot(data_dict:Dict):
        '''
        提取炮号
        '''
        if not ('shot' in data_dict):
            try:
                shot = re.search(r'/(\d+)\.csv$', annotations['csv']).group(1)
            except:
                print('未读取到炮号')
                return None        
        else:
            shot = data_dict['shot']
        return shot
                
    label_list = []
    shot_set = set()
    # 对应pgsql的列名
    if (not multi_label_group) and label_group_name is None:
        label_group_name = 'label'
    # json_min格式    
    if json_min:
        # 非标注内容的key
        common_keys = {"csv", "shot", "id",
                   "annotator", "annotation_id",
                   "created_at", "updated_at", "lead_time"}
        for annotations in annotation_json:
                # 得到可能的标注
                label_keys = set(annotations.keys())-common_keys
                label_data = {}
                # 获取shot
                shot = get_shot(annotations)
                if not shot:
                    continue
                # 如果只允许一炮有一个标注，筛除重复的标注    
                if shot_unique and (shot in shot_set):
                    print(f'炮号 {shot} 已存在')
                    continue
                shot_set.add(shot)
                base = dict(
                    shot=shot,
                    annotator=annotation["annotator"],
                    annotation_id=annotation["id"],
                    annotation_created=annotation["created_at"],
                    annotation_updated=annotation["updated_at"],
                    )
                for key in label_keys:
                    for label_result in annotations[key]:
                        label_data = base.copy()
                        try:
                            if multi_label_group:
                                label_data['feature'] = key
                                if not only_one_choice:
                                    label_data['label'] = label_result['timeserieslabels'][0]
                            elif not only_one_choice:    
                                label_data[label_group_name] = label_result['timeserieslabels'][0]
                            label_data['start_time'] =  label_result['start']
                            label_data['end_time'] =  label_result['end']
                            label_list.append(label_data)
                        except:
                            print('读取标注错误，意料之外的内容')
    # json格式
    else:
        # full json以task为单位
        for task in annotation_json:
                # 获取炮号
                shot = get_shot(task['data'])
                if shot is None:
                    continue
                # 跳过重复的炮
                if shot_unique and (shot in shot_set):
                    print(f'炮号 {shot} 已存在')
                    continue
                label_data = {}
                shot_set.add(shot)
                annotations = task['annotations']
                # 检查该任务是否有标注
                if annotations == []:
                    print(f'{shot} 无标注结果')
                    if include_no_annotation:
                        label_list.append(label_data)    
                    continue
                for annotation in annotations:
                     # 获取具体的标注内容
                    base = dict(
                        shot=shot,
                        annotator=annotation.get("completed_by",0),
                        annotation_id=annotation["id"],
                        annotation_created=annotation["created_at"],
                        annotation_updated=annotation["updated_at"],
                    )
                    # 处理skipped标注
                    if taking_skipped_as_label and annotation.get("was_cancelled",False):
                        label_data = base.copy()
                        label_data[label_group_name] = 'skipped'
                        label_list.append(label_data) 
                    for label_result in annotation['result']:
                        label_data = base.copy()
                        label_data['start_time'] = label_result['value']['start']
                        label_data['end_time'] = label_result['value']['end']
                        if multi_label_group:
                            label_data['feature'] = label_result['from_name']
                            if not only_one_choice:
                                label_data['label'] = label_result['value']['timeserieslabels'][0]
                        elif not only_one_choice:
                            label_data[label_group_name] = label_result['value']['timeserieslabels'][0]
                        label_list.append(label_data)
                    if shot_unique:
                        break    
    return label_list

def image_json_convertor(annotation_json, 
                         json_min :bool = True, 
                         image_unique = False, 
                         to_real_pixel:bool = True,
                         aggregate_numbers: bool = True, 
                         multi_label_group :bool = False, 
                         include_no_annotation :bool = False,
                         taking_skipped_as_label:bool = False):
    """
    将Label Studio导出的图像标注数据进一步化简，用于将数据保存至pgsql
    该函数只适用于label studio社区版，企业版的格式可能会有一定变化
    
    参数:
    ----
    annotation_json: dict
        export_annotation返回结果，或者其他以字典格式存储的json_min或json的标注数据 
    json_min : bool
        json_min为True则annotation_json为json_min语法，否则annotation_json为完整的json语法
    image_unique: bool
        是否只允许一张图片有一组标注结果，若multi_label_group为True或一个标注有多个mask，仍可能返回同属一个标注者的一组多个标注结果
    aggregate_numbers: bool
        是否将一组标注中的所有Number聚合在一起形成一条标注。默认为True    
    multi_label_group : bool
        是否存在多组标签
    include_no_annotation: bool
        是否导出无标注的任务，仅在json_min=False时有效。若include_no_annotation=True，则包含无标注任务的图片
    taking_skipped_as_label: bool
        是否将skipped也作为一种label，仅在json_min=False时有效。若taking_skipped_as_label=True，则将被跳过作为标签加入解析结果。
    返回:
    ----
    label_list: List[dict]
        简化的标注结果，用于插入Postgresql数据库的形式
    """
                
    label_list = []
    image_set = set()
    # 对应pgsql的列名
    #if (not multi_label_group) and label_group_name is None:
    #    label_group_name = 'label'
    # json_min格式    
    if json_min:
        # 非标注内容的key
        common_keys = {"image", "shot", "frame_time", "id",
                   "annotator", "annotation_id",
                   "created_at", "updated_at", "lead_time"}
        for annotation in annotation_json:
                # 得到可能的标注
                label_keys = set(annotation.keys())-common_keys
                # 获取shot和帧物理时间
                shot, frame_time = _get_shot_time(annotation)
                if shot is None:
                    continue
                # 如果图片唯一，跳过重复的图片
                if image_unique and (shot, frame_time) in image_set:
                    print(f'炮号 {shot} {frame_time}s 的图片已存在')
                    continue
                image_set.add((shot, frame_time))
                base = dict(
                    shot=shot, frame_time=frame_time,
                    annotator=annotation["annotator"],
                    annotation_id=annotation["id"],
                    annotation_created=annotation["created_at"],
                    annotation_updated=annotation["updated_at"],
                    )
                number_dict = {} #用于聚合Number标注
                for key in label_keys:
                    for label_result in annotation[key]:
                        if isinstance(label_result,str):
                            print(f'skipped label:{label_result}')
                            break
                        label_data = base.copy()
                        try:
                            if multi_label_group:
                                label_data['feature'] = key
                            label_data['image_width'] =  label_result.get('original_width', 1280)
                            label_data['image_height'] =  label_result.get('original_height', 800)
                            if 'number' in label_result :
                                if aggregate_numbers:
                                    number_dict[key] = label_result['number']
                                    continue
                                else:
                                    geo = parse_number(label_result)
                                    geo.update({'label':key})
                            else:        
                                geo = _extract_geometry(label_result, to_real_pixel, (label_data['image_height'],label_data['image_width']))
                            if geo:
                                label_data.update(geo)       
                                label_list.append(label_data)            
                        except Exception as e:
                            print(f'读取标注错误: {e}')
                if number_dict:
                    label_data = base.copy()
                    label_data.update({'geom_data':number_dict,
                                       'geom_type':'number'})     
                    label_list.append(label_data)       
    # json格式
    else:
        # full json以task为单位
        for task in annotation_json:
                source_data = task['data']
                # 获取炮号和帧物理时间
                shot, frame_time = _get_shot_time(source_data)
                if shot is None:
                    continue
                # 跳过重复的图片
                if image_unique and (shot, frame_time) in image_set:
                    print(f'炮号 {shot} {frame_time}s 的图片已存在')
                    continue
                image_set.add((shot, frame_time))
                annotations = task['annotations']
                # 检查该任务是否有标注
                if annotations == []:
                    print(f'{shot}: {frame_time}s 的图片无标注结果')
                    if include_no_annotation:
                        label_list.append({'shot':shot, 'frame_time':frame_time})    
                    continue
                for annotation in annotations:
                    number_dict = {} #用于聚合Number标注
                     # 获取具体的标注内容
                    base = dict(
                        shot=shot, frame_time=frame_time,
                        annotator=annotation.get("completed_by",0),
                        annotation_id=annotation["id"],
                        annotation_created=annotation["created_at"],
                        annotation_updated=annotation["updated_at"],
                    )
                    # 处理skipped标注
                    if taking_skipped_as_label and annotation.get("was_cancelled",False):
                        label_data = base.copy()
                        label_data['label'] = 'skipped'
                        label_list.append(label_data)
                    for label_result in annotation['result']:
                        label_data = base.copy()
                        label_data['image_width'] = label_result.get('original_width', 1280)
                        label_data['image_height'] = label_result.get('original_height', 800)
                        if label_result['type'] =='number':
                            if aggregate_numbers:
                                number_dict[label_result['from_name']] = label_result['value']['number']
                                continue
                            else:
                                geo = parse_number(label_result['value'])
                                geo.update({'label':label_result['from_name']})
                        else:        
                            geo = _extract_geometry(label_result['value'], to_real_pixel, (label_data['image_height'],label_data['image_width']))
                        if geo:
                            label_data.update(geo)
                            if multi_label_group:
                                label_data['feature'] = label_result['from_name']
                            label_list.append(label_data)
                    if number_dict:
                        label_data = base.copy()
                        label_data.update({'geom_type':'number','geom_data':number_dict})
                        label_list.append(label_data)        
                    if image_unique:
                        break    
    return label_list

# ------导出时序数据及图像数据------
def _remove_used_channels(processed_data, raw_data_columns) -> None:
        """
        从 processed_data 中解析所有 (数据别名[channel]) 出现过的通道，并将这些通道
        从 raw_data_columns 的通道列表里删除。
        """

        # 用正则匹配形如  data_alias[channel]  的语法
        pattern = r'([a-zA-Z_]\w*)\[([\w]+)\]'
        raw_data_columns
        for item in processed_data:
            expr = item.get("expression", "")
            # 在 expression 里查找所有匹配
            matches = re.findall(pattern, expr)
            # matches : list of (data_alias, channel)
            for (data_alias, channel) in matches:
                # 如果 raw_data_map 中存在这个 data_alias，就删除相应的 channel
                if data_alias in raw_data_columns.keys():
                    table_name, channel_list = raw_data_columns[data_alias]
                    # 转换为字符串，便于比较
                    channel_list = list(map(str, channel_list))
                    if channel in channel_list:
                        channel_list.remove(channel)        
                    if not channel_list:
                        del raw_data_columns[data_alias]
                    else:
                        raw_data_columns[data_alias] = (table_name, channel_list)                    
        return raw_data_columns 

def create_data_view(shots:List[int], 
                     view_name:str, 
                     raw_data_columns:dict, 
                     processed_data: List[dict],
                     including_raw_data: bool,
                     keep_used_raw_data: bool)->dict:
    '''
    在给定的炮号上，基于数据需求创建View以导入数据

    参数：
    ----
    shots: List[int]
        炮号列表
    view_name: str
        视图名称，推荐使用项目名称
    raw_data_columns: dict
        由原始数据的表和表头组成的字典
    processed_data: List[dict]
        新数据的列表，每一个新的数据通道需要以特定的形式表示
        [{name: "ip", 
        expression: "ip0[of1]-ip_eddy[of1]"}]
    including_raw_data: bool
        是否保留使用过的原始数据,默认为false
    keep_used_raw_data: bool
        是否将所有数据保存在视图中。若为true，创建的视图会包括所有数据,默认为false

    返回：
    ----
    data_column: dict
        包括View在内的，用于prepare_data以及create_project的数据表    
    '''
    if processed_data is {}:
        return raw_data_columns
    utils.delete_view(shots, view_name)
    view_columns = utils.create_view(shots, 
                                        raw_data_columns, 
                                        processed_data, 
                                        view_name, 
                                        including_raw_data, keep_used_raw_data)
    view_dict = {'view_data': (f"{view_name}", view_columns)}
    if including_raw_data:
        return view_dict
    if not keep_used_raw_data:
        raw_data_columns = _remove_used_channels(processed_data,
                                                 raw_data_columns)
    raw_data_columns.update(view_dict)
    return raw_data_columns

def prepare_data(project_name:str, 
                 shots:List[int]|int, 
                 name_table_columns:dict, 
                 t_min: Optional[float]=None, 
                 t_max: Optional[float]=None, 
                 resolution: Optional[float]=None,
                 filters: Optional[List]=None):
    """
    准备数据。从pgsql数据库中读取数据为csv文件，并生成对应的URL
    
    参数:
    ----
    project_name: str
        项目名称
    shot: List[int]|int
        炮号(可为数字或字符串)，也可以是包含多个炮号的可迭代对象。    
    name_table_columns: dict
        形如:
        {
            "some_name": ("table_name", ["col1", "col2", ...]),
            "other_name": ("other_table", ["colA", "colB", ...])
        }
        字典的key随意，但value中需要指定数据库的表名和对应列名。
    t_min: float, 可选
        数据的最小时间
    t_max: float, 可选
        数据的最大时间
    resolution: float, 可选
        数据降采样分辨率(秒)
    
    返回:
    ----
    urls
        返回一个字典，每个键为炮号，值为对应数据的url。
    """
    # 从pgsql导出数据，生成URLs
    result = utils.export_data(project_name, shots, name_table_columns, t_min, t_max, resolution)
    if result.get('error',None):
        print(result["error"])
        return {}
    else:
        urls = result["urls"]
    if not filters:
        urls, _ = utils.filter_data(urls, filters, True, dir=project_name)
    # 写入Redis数据库
    result = utils.export_urls_to_redis(project_name=project_name, urls=urls, key="csv")
    return urls


def prepare_imgs(project_name: str,
                 shots: list[int]|int,
                 t_min: float|None = None,
                 t_max: float|None = None,
                 resolution: float|None = None):
    """
    从视频中提取图像并写入 Redis
    
    参数:
    ----
    project_name: str
        项目名称
    shot: List[int]|int
        炮号(可为数字或字符串)，也可以是包含多个炮号的可迭代对象。    
    t_min: float, 可选
        数据的最小时间
    t_max: float, 可选
        数据的最大时间
    resolution: float, 可选
        数据降采样分辨率(秒)
    
    返回:
    ----
    urls: dict
        存储图片url，每个键为炮号，值为对应数据的url列表。
    frame_time: dict
        存储图片的帧时间，每个键为炮号，值为对应图片的帧时间    
    """
    result = utils.export_imgs(project_name, shots, t_min, t_max, resolution)
    if result.get("error", None):
        print(result["error"])
        return {}

    urls = result["urls"] #图像数据中，一个shot对应一组url
    frame_times = result['frame_time']
    utils.export_urls_to_redis(project_name, urls, key="image", frame_time=frame_times)
    return (urls, frame_times)

# ------导入预标注------
def import_prediction_timeseries(
        project_name :str, 
        predictor : BasePredictor, 
        urls: Dict[int, str], 
        model_version: Optional[str] = 'prelabel_script',
        **extra_args
        ):
    '''
    根据url导入预标注结果

    参数:
    ----
    project_name: str
        Redis路径，一般为项目名称
    predictor: prediction.BasePredictor
        用于生成预标注
    urls: (Dict[int, str]): 
        数据 URL 列表
    model_version: str
        模型版本
    extra_args
        predictor可能需要的额外参数    
    '''
    
    data = utils.load_data(urls)
    conn = utils.connect_redis()
    pattern = f"{project_name}/*"
    keys = set(conn.keys(pattern=pattern))
    for shot, df in data.items():
        preds = []
        ctx = prediction.DataContext(shot=shot)
        try:
            preds = to_labelstudio_form(predictor.predict(df, data_context=ctx, **extra_args), 
                                        model_version)
            print(f'{shot}预标注成功')
        except Exception as e:
            print(f'生成{shot}的预标注失败：{e}')
            continue
        key = f'{project_name}/{shot}'
        if key in keys:
            task_str = conn.get(key)
            try:
                task = json.loads(task_str)
            except json.JSONDecodeError as e:
                print(f"解析 JSON 数据时出错:{e},将重写 {key} 数据")
                task = {"data": 
                        {"csv": urls[shot],"shot": shot}} 
            task['predictions'] = preds
            conn.set(key, json.dumps(task, ensure_ascii = False))
        else:
            data_dict ={'csv' : urls[shot], "shot" : shot}
            conn.set(key, json.dumps({"data": data_dict, 'predictions':preds}, ensure_ascii=False))    
    conn.close()        

def import_prediction_image(project_name :str, 
                            predictor : BasePredictor, 
                            urls: Dict[int, List[str]], 
                            frame_times: Dict[int, List[float]]|None = None, 
                            model_version: Optional[str] = 'prelabel_script',
                            **extra_args):
    '''
    根据url导入预标注结果

    参数:
    ----
    project_name: str
        Redis路径，一般为项目名称
    predictor: prediction.BasePredictor
        用于生成预标注
    urls: Dict[int, List[str]]
        图像 URL 列表
    frame_times: Dict[int, List[float]]]
        图像 帧时间 列表 
    model_version: str
        模型版本
    extra_args
        predictor可能需要的额外参数
    '''
    
    img_dict = utils.load_imgs(urls)
    conn = utils.connect_redis()
    pattern = f"{project_name}/*"
    keys = set(conn.keys(pattern=pattern))
    for shot, images in img_dict.items():
        # 获取图片的帧时间，用于在Redis数据库中读取和存储图片
        if frame_times and shot in frame_times:
            times = frame_times[shot]
        else:
           times = utils.parse_urls_frame_times(urls[shot])
        if times is None or len(times) != len(images):
            print(f" Shot {shot} 帧时间缺失或数量不匹配，将以索引代替")
            times = list(range(len(images)))   
        for idx, (img, ft) in enumerate(zip(images, times)):
            preds = []
            ctx = prediction.DataContext(shot=shot, frame_time=ft)
            try:
                preds = to_labelstudio_form(predictor.predict(img, data_context=ctx, **extra_args), 
                                            model_version)
                print(f'{shot}: {ft}s 预标注成功')
            except Exception as e:
                print(f'生成{shot}: {ft}s 的预标注失败：{e}')
                continue
            suffix = int(round(ft * 1e5))
            key = f'{project_name}/{shot}_{suffix}'
            if key in keys:
                task_str = conn.get(key)
                try:
                    task = json.loads(task_str)
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 数据时出错:{e},将重写 {key} 数据")
                    task = {"data": 
                            {"image": urls[shot][idx],"shot": shot,"frame_time": ft,
                        }}     
                task['predictions'] = preds
                conn.set(key, json.dumps(task, ensure_ascii = False))
            else:
                data_dict ={'image' : urls[shot][idx], "shot" : shot, "frame_time":ft}
                conn.set(key, json.dumps({"data": data_dict, 'predictions':preds},ensure_ascii=False))    
    conn.close()

# ------从Label Studio加载原始数据------

def load_project_data(ls, project_id:int, url_only :bool = True, exclude_skipped=True, only_with_annotation=True):
    '''
    返回一个项目中所选任务的数据url或原始数据

    参数:
    ----
    ls: Label Studio客户端
    project_id: int
        项目编号
    url_only: True
        是否导出任务原始数据，True则只导出url。默认为True    
    exclude_skipped: bool
        是否包括在Label Studio中被跳过的任务。True则包括跳过的任务。默认为True。    
    only_with_annotation: bool
        only_with_annotation=True则只导出有标注的数据,否则也导出无标注的数据。默认为True                       
    返回:
    ----
    urls: dict
        json或者json-min格式的标注数据

    '''
    import re
    export_json = export_annotation(ls, project_id, json_min=True, 
                                    exclude_skipped=exclude_skipped, only_with_annotation=only_with_annotation)
    urls = {}
    for data_dict in export_json:
        # data = data_dict['data']
        if 'shot' in data_dict:
            urls[data_dict['shot']] = data_dict['csv']
        else:
            shot_num = int(re.search(r'/(\d+)\.csv$', data_dict['csv']).group(1))
            urls[shot_num] = data_dict['csv']
    if url_only:
        return urls
    else:
        raw_data = utils.load_data(urls)
        return urls, raw_data        


def load_project_imgs(ls, project_id:int, url_only :bool = True, exclude_skipped=True, only_with_annotation=True):
    '''
    返回一个项目中所选任务的图像数据url及对应帧时间或原始图像

    参数:
    ----
    ls: Label Studio客户端
    project_id: int
        项目编号
    url_only: True
        是否导出任务原始数据，True则只导出url。默认为True    
    exclude_skipped: bool
        是否包括在Label Studio中被跳过的任务。True则包括跳过的任务。默认为True。    
    only_with_annotation: bool
        only_with_annotation=True则只导出有标注的数据,否则也导出无标注的数据。默认为True                       
    返回:
    ----
    urls: dict
        json或者json-min格式的标注数据

    '''
    export_json = export_annotation(
        ls, project_id,
        json_min=True,
        exclude_skipped=exclude_skipped,
        only_with_annotation=only_with_annotation
    )

    urls:  Dict[int, List[str]]   = defaultdict(list)
    frame_times: Dict[int, List[float]] = defaultdict(list)

    # 收集数据
    for ann in export_json:
        shot, frame_time = _get_shot_time(ann)    
        urls[shot].append(ann["image"])
        frame_times[shot].append(frame_time)

    # 每个 shot 内部按时间排序
    for shot in urls:
        pairs = sorted(zip(frame_times[shot], urls[shot]))
        frame_times[shot], urls[shot] = map(list, zip(*pairs))

    if url_only:
        return urls, frame_times      
    else:
        img_dict = utils.load_imgs(urls)   
        return (urls, frame_times), img_dict

# ------Label Studio存储管理 ------

def list_project_storage(ls, project_id: int):
    """
    返回某个项目下的所有本地redis存储

    参数:
    ----
    ls: Label Studio客户端
    project_id: int
        项目序号
    
    返回:
    ----
    storage_list: list
        返回由redis储存组成的列表
    """
    storage_list = ls.import_storage.redis.list(project=project_id)
    for storage in storage_list:
        print([f'id={storage.id}', f'title={storage.title}', f'path={storage.path}', f'regex_filter={storage.regex_filter}'])
    return storage_list

def create_storage(ls, project_id:int, project_name:str=None, path :Optional[str]= None, title=None, description=None, regex_filter=None):
    """
    创建、验证并同步Redis储存
    
    参数:
    ----
    ls: object
        连接Label Studio客户端后返回的变量
    project_id: int
        项目编号
    project_name: str
        项目标题，默认为None。若提供project_name请保证与project_id对应的项目一致
    path: str
        Redis的存储路径    
    title: str
        可选。本地存储标题。
    description: str
        可选。本地存储描述。
    regex_filter: str
        可选。正则表达式过滤器。当json_import=False时，可以使用该正则表达式筛选文件。
    返回:
    ----
    storage
        创建的Redis存储的信息
    """
    if project_name is None:
        project = ls.projects.get(project_id)
        project_name = project.title
    if title is None:
        date = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        title = f'{project_name}_storage_{date}'
    if description is None:
        description = title
    if path is None:
        path = project_name
    if regex_filter is not None:        
        storage = ls.import_storage.redis.create(project=project_id, title=title, description=description,
                                    path = path, use_blob_urls=False, regex_filter = regex_filter,
                                    host = config.REDIS_HOST, port = config.REDIS_PORT, password = config.REDIS_PASSWORD)
    else:
        storage = ls.import_storage.redis.create(project=project_id, title=title, description=description,
                                    path = path, use_blob_urls=False,
                                    host = config.REDIS_HOST, port = config.REDIS_PORT, password = config.REDIS_PASSWORD)
    print(storage)        
    sync_response = sync_storage(ls, project_id = storage.project, storage=storage)       
    return storage


def sync_storage(ls, project_id:int, storage):
    '''
    同步已经创建的redis储存到Label Studio

    参数:
    ----
    ls: object
        连接Label Studio客户端后返回的变量
    project_id: int
        标识项目的唯一变量    
    storage: int或Redis储存
        已有本地存储或本地储存的id
    返回:
    ----
    sync_response: dict
        同步结果
    '''
    if isinstance(storage, int):
        storage = ls.import_storage.redis.get(id = storage)
    # 该部分存在bug，UI界面和API可以同步，但是API中验证储存无法通过
    #try:
        #val_response = ls.import_storage.redis.validate(id = storage.id, project=project_id,
        #                                                title = storage.title, description=storage.description,
        #                                                regex_filter=storage.regex_filter, use_blob_urls=storage.use_blob_urls,
        #                                                path = storage.path, host = storage.host, 
        #                                                port = storage.port, password = storage.password)
    #except Exception as e:
    #    print(f'Redis储存验证失败：{e}') 
    sync_response = ls.import_storage.redis.sync(id = storage.id)
    return sync_response


def delete_storage(ls, storage_id : int, keep_storage_link : bool = False):
    """
    删除存储相关的redis中的文件以及原始文件(img或csv)
    
    参数:
    ----
    ls: Label Studio客户端
    storage_id: int
        存储编号
    keep_storage_link: bool
        是否在 Label Studio 中保留redis存储记录
    """
    # 获取存储路径
    storage = ls.import_storage.local.get(storage_id)
    path = storage.path
    # 删除redis中的数据
    utils.delete_redis_file(path)     
    # 删除csv文件
    response = utils.delete_file(path)
    print(response)
    # 如果 keep_storage_link=False，则删除LS里的存储记录
    if not keep_storage_link:
        ls.import_storage.local.delete(storage_id)
        print(f"已删除 Storage ID={storage_id}。")

