import json
import yaml
from . import utils, prediction
import re
from typing import Dict, List, Callable, Optional, Any
import pandas as pd
from . import toklabel

class ProjectBuilder():
    project: str
    project_description: Optional[str]
    project_id: Optional[int]
    project_data_type: str
    shots: int|List[int]|Dict
    raw_data: Dict[str, Any]
    t_min: Optional[float|list]
    t_max: Optional[float|list]
    resolution: Optional[float]
    file_path: str
    processed_data: List[Any]
    keep_used_raw_data: bool
    including_raw_data: bool
    filter: List[str]
    data_exported: bool
    label_config: List[Any]
    using_SAM2: bool
    using_TimeSeries_filter: bool
    
    def __init__(self, project_config_file: str, **kwargs):
        defaults = {
            "project": "default_project",
            "project_description": "",
            "project_id": None,
            "project_data_type": "Timeseries",
            "shots": [],
            "raw_data": {},
            "t_min": None,
            "t_max": None,
            "resolution": 1e-4,
            "file_path":None,
            "processed_data": [],
            "keep_used_raw_data": True,
            "including_raw_data": False,
            "filter": [],
            "label_config": [],
            "data_exported": False,
            "using_SAM2": True,
            "using_TimeSeries_filter": False
        }
        # 从 yaml 文件中加载默认配置
        try:
            with open(project_config_file, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
        except FileNotFoundError:
            print('配置文件不存在')
            file_config = {}

        raw_data = file_config.get("raw_data", {})
        file_config["raw_data"] = self._parse_raw_data(raw_data) if raw_data else {}
        #labels = file_config.get("label_config", [])
        #file_config["label_config"] = self._parse_label_config(labels) if labels else []
        filter_expr = file_config.get("filter", [])
        file_config["filter"] = self._parse_filter_expression(filter_expr) if filter_expr else []

        defaults.update(file_config)
        defaults.update(kwargs)

        self.__dict__.update(defaults)
        # 项目实际输入的数据格式
        self.project_input_data = self.raw_data
        self.existed_project = True if self.project_id else False
        if not self.file_path:
            self.file_path = self.project

    def _parse_raw_data(self, raw_data:dict):
        '''
        转换raw_data数据格式
        '''
        raw_data_map = {}
        for data_alias, info in raw_data.items():
            table_name = info["data_table_name"]
            channel_list = info["channels"]
            raw_data_map[data_alias] = (f"{table_name}", [str(c) for c in channel_list])
        return raw_data_map
    
    def _parse_label_config(self, labels:List):
        '''
        根据配置文件生成label_config
        '''
        label_config = []
        for label in labels:
            label_config.append((label["label_group"], 
                                 label['label_choices']))
        return label_config    

    def _convert_channel_syntax(self, expr: str) -> str:
        """
        ip0[of1] -> ip0_of1
        some[123] -> some_123
        """
        # 匹配  name[...]  => name_...
        pattern = r'([a-zA-Z_]\w*)\[([\w]+)\]'
        repl = r'\1_\2'
        return re.sub(pattern, repl, expr)

    def _parse_filter_expression(self, exprs: str|List[str]) -> List[Callable[[pd.DataFrame], bool]]:
        """
        将形如 'max(ip)>100e3' 的字符串
        解析为一个可对 DataFrame 执行检查的函数 f(df)->bool。
        
        目前仅支持:
        max(...) < value
        max(...) > value
        min(...) < value
        min(...) > value

        返回
        - lambda 函数，处理Pandas.DataFrame数据，符合要求返回True，否则返回bool
        
        """
        # 1) 转换  flux_loop[2] => flux_loop_2
        if isinstance(exprs, str):
            exprs = [exprs]
        filter_func = []    
        for expr in exprs:
            if not isinstance(expr, str):
                print(f'{expr} is not str, is {type(expr)}')
                continue
            expr_converted = self._convert_channel_syntax(expr)

            # 2) 用正则提取  (min|max)(column) (operator) (number)
            pattern = r'^\s*(min|max)\(([\w_]+)\)\s*([<>])\s*([\-\+\w\.]+)\s*$'
            m = re.match(pattern, expr_converted)
            if not m:
                raise ValueError(f"无法解析filter表达式: {expr}")

            func_name = m.group(1)  # "min" or "max"
            col_name  = m.group(2)  # e.g. "ip", "flux_loop_2"
            operator  = m.group(3)  # '<' or '>'
            thresh_str= m.group(4)  # "100e3", "1.5e2" etc.

            # 转换阈值为 float
            try:
                threshold = float(thresh_str)
            except ValueError as e:
                raise ValueError(f"阈值无法转换为数值: {expr}, {e}")

            # 3) 构造对 DataFrame 的检查函数
            if func_name == 'min':
                if operator == '<':
                    filter_func.append(lambda df: df[col_name].min() < threshold)
                elif operator == '>':
                    filter_func.append(lambda df: df[col_name].min() > threshold)
                else:
                    raise ValueError(f"不支持的运算符: {operator}")
            elif func_name == 'max':
                if operator == '<':
                    filter_func.append(lambda df: df[col_name].max() < threshold)
                elif operator == '>':
                    filter_func.append(lambda df: df[col_name].max() > threshold)
                else:
                    raise ValueError(f"不支持的运算符: {operator}")
            else:
                raise ValueError(f"不支持的函数: {func_name}")
        return filter_func

    def generate_shot_list(self, shots=None):
        '''
        生成炮号列表
        '''
        if shots is None:
            shots = self.shots
        if isinstance(shots, (str, int)):
            return [shots]
        elif isinstance(shots, dict):
            try:
                return utils.shot_range(shots['min'], shots['max'])
            except Exception as error:
                print(f'生成 shots_list 错误：{error}')
                return []
        elif isinstance(shots, list):
            return shots
        else:
            raise ValueError('unexcepted type of variable in ProjectBuilder.shots')
        
    def create_view(self):
        '''
        生成视图VIEW
        '''
        if self.processed_data is {}:
            return []
        shots = self.generate_shot_list()
        utils.delete_view(shots, self.project)
        view_columns = utils.create_view(shots, 
                                          self.raw_data, 
                                          self.processed_data, 
                                          self.project, 
                                          self.including_raw_data, self.keep_used_raw_data)
        return view_columns
    
    
    def update_project_data_config(self, view_columns:List):
        '''
        根据视图，更新项目输入数据
        '''
        if view_columns is []:
            return self.project_input_data
        # 来自视图的数据，在name_table_column中，key都为'view_data'
        view_dict = {'view_data': (f"{self.project}", view_columns)}
        if self.including_raw_data:
            self.project_input_data = view_dict
            return self.project_input_data
        if not self.keep_used_raw_data:
            self.project_input_data = self.remove_used_channels()
                 
        self.project_input_data.update(view_dict)
        return self.project_input_data


    def remove_used_channels(self) -> None:
        """
        从 processed_data 中解析所有 (数据别名[channel]) 出现过的通道，并将这些通道
        从 raw_data_map 的通道列表里删除。
        """

        # 用正则匹配形如  data_alias[channel]  的语法
        pattern = r'([a-zA-Z_]\w*)\[([\w]+)\]'
        processed_data = self.processed_data
        raw_data_map = self.raw_data
        for item in processed_data:
            expr = item.get("expression", "")
            # 在 expression 里查找所有匹配
            matches = re.findall(pattern, expr)
            # matches : list of (data_alias, channel)
            for (data_alias, channel) in matches:
                # 如果 raw_data_map 中存在这个 data_alias，就删除相应的 channel
                if data_alias in raw_data_map.keys():
                    table_name, channel_list = raw_data_map[data_alias]
                    # 转换为字符串，便于比较
                    channel_list = list(map(str, channel_list))
                    if channel in channel_list:
                        channel_list.remove(channel)        
                    if not channel_list:
                        del raw_data_map[data_alias]
                    else:
                        raw_data_map[data_alias] = (table_name, channel_list)                    
        return raw_data_map                

    def prepare_data(self, shots = None):
        '''
        使用ProjectBuilder快速准备项目数据

        参数:
        ----
        shots: int|dict|List[int]
            炮号，当不传入参数时，默认使用project_builder内存储的炮号。
            以dict形式传入参数，需要包含"min"和"max"两个key，代表需要数据的炮号范围

        返回：
        ----
        urls: dict
            返回一个字典，每个键为炮号，值为对应数据的url。  
        '''
        view_channels = self.create_view()
        self.project_input_data = self.update_project_data_config(view_channels)
        shots = self.generate_shot_list(shots)
        urls = toklabel.prepare_data(self.file_path, shots, self.project_input_data, 
                        self.t_min, self.t_max, self.resolution, self.file_path, self.filter)
        if urls:
            self.data_exported = True
            return urls
        else:
            return {}

    def prepare_img(self, shots = None):
        '''
        使用ProjectBuilder快速准备图像数据

        参数:
        ----
        shots: int|dict|List[int]
            炮号，当不传入参数时，默认使用project_builder内存储的炮号。
            以dict形式传入参数，需要包含"min"和"max"两个key，代表需要数据的炮号范围

        返回：
        ----
        urls: dict
            返回一个字典，每个键为炮号，值为对应图像的url列表。
        '''
        shots = self.generate_shot_list(shots)
        if self.using_TimeSeries_filter and self.filter:
            shots = self.prepare_data(shots).keys()   
            utils.delete_redis_file(self.file_path)
            utils.delete_file(self.file_path)
        if shots:
            self.data_exported = True     
            return toklabel.prepare_imgs(self.project,
                                shots,
                                self.t_min,
                                self.t_max,
                                self.resolution)
        else:  
            print('输入炮号无效或所有炮号的时序数据未通过filter')
            return {}
            

    def create_project(self, ls):
        '''
        使用ProjectBuilder快速创建项目

        参数:
        ----
        project_builder: toklabel.ProjectBuilder
            存储项目输入数据配置信息的实例
        ls : label Studio客户端

        返回：
        ----
        urls: dict
            返回一个字典，每个键为炮号，值为对应数据的url。  
        '''
        if self.existed_project or self.project_id:
            raise Exception(f"project already existed, whose id is {self.project_id}. If you want to create project again, use create_existed_project instead")
        proj = toklabel.create_project(ls=ls, 
                                       project_name=self.project, 
                                       data_type=self.project_data_type, 
                                       description=self.project_description, 
                                       name_table_columns=self.project_input_data, 
                                       label_groups=self.label_config,
                                       using_SAM_support=self.using_SAM2)
        self.existed_project = True
        self.project_id = proj.id
        return proj
    
    def create_existed_project(self, ls):
        '''
        使用ProjectBuilder再创建一个已经存在的项目

        参数:
        ----
        project_builder: toklabel.ProjectBuilder
            存储项目输入数据配置信息的实例
        ls : label Studio客户端

        返回：
        ----
        urls: dict
            返回一个字典，每个键为炮号，值为对应数据的url。  
        '''
        xml_config = toklabel.create_timeseries_label_config(self.project_input_data, self.label_config)
        proj = ls.projects.create(
            title= self.project,
            description= self.project_description,
            label_config=xml_config,
            show_skip_button=True,
            show_instruction=True,
            show_annotation_history=True
        )
        self.existed_project = True
        self.project_id = proj.id
        return proj
    
    def import_prediction(self, predictor: prediction._LegacyBasePredictor, urls:Optional[Dict[int, str]] = {}, model_version:Optional[str] = 'prelabel_script'):
        '''
        使用ProjectBuilder导入预标注

        参数：
        ----
        project_name: str
            Redis路径，一般为项目名称
        predictor: prediction.BasePredictor
            用于生成预标注
        urls: (Dict[int, str]): 
            数据 URL 列表， 如果为空，则使用 ProjectBuilder 内部记录的shot生成urls
        model_version: str
            模型版本
    
        '''
        if not urls:
            if self.data_exported:
                url_list = utils.list_files(self.file_path)
                urls = utils.parse_urls_shots(url_list) if isinstance(url_list, list) else {}
            if not urls:
                urls = self.prepare_data()
        toklabel.import_prediction_timeseries(self.project, predictor, urls, model_version)



    def create_storage(self, ls, title:str=None, description:str=None):
        '''
        使用ProjectBuilder创建、验证并同步Redis储存
    
        参数:
        ----
        ls: object
            连接Label Studio客户端后返回的变量
        project_id: int
            项目编号
        title: str
            可选。本地存储标题。
        description: str
            可选。本地存储描述。
        返回:
        ----
        storage
            创建的Redis存储的信息
        '''
        return toklabel.create_storage(ls, self.project_id, project_name=self.project, path=self.file_path, title=title, description=description)

if __name__ == "__main__":
    pb = ProjectBuilder('project_config.json', t_max=5)
    print(pb.t_max)
    print(pb.raw_data)
    sql = utils.generate_create_view_sql(pb.raw_data, pb.processed_data, pb.project)
    print(sql)