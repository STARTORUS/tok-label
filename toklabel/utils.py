from .config import FILE_SERVER_URL, EXTRACTOR_SERVER_URL, REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB
import requests
from typing import Dict, List, Optional, Tuple, Callable, Union
import redis
import json
from io import BytesIO
import pandas as pd
import os
import sunist2.script.postgres as pg
from sunist2.script.postgres import (
    connect_to_database,
    execute_query,
    close_connection
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import re
import psycopg2
import cv2
import numpy as np
from label_studio_sdk.converter import brush as rle_util

import logging
logger = logging.getLogger(__name__)

def list_files(dir: str, recursive: bool = False):
    """
    列出目录下的所有文件

    参数:
    dir (str): 目录路径
    recursive(bool): 是否递归查询子目录

    返回:
    List[str]: 列表，每个元素为文件的 URL
    """
    try:
        params = {"dir": dir, "recursive": recursive} 
        response = requests.get(f"{FILE_SERVER_URL}/list/", params=params)
        response.raise_for_status()
        # add url to files
        result = response.json()
        if "urls" in result:
            result["urls"] = [f"{FILE_SERVER_URL}{url}" for url in result["urls"]]
        return result
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def delete_file(dir: str, file: str=None):
    """
    删除文件

    参数:
    dir (str): 目录路径
    file (str, optional): 文件名

    返回:
    Dict: 包含操作结果的字典
    """
    try:
        params = {"dir": dir, "file": file} if file else {"dir": dir}
        response = requests.delete(f"{FILE_SERVER_URL}/delete/", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
def export_data(
        project_name: str, 
        shots: List[int], 
        name_table_columns: Dict, 
        t_min: Optional[float]=None, t_max: Optional[float]=None, 
        resolution: Optional[float]=None
        ):
    """
    导出数据

    参数:
    project_name (str): 项目名称, 也是文件夹名称
    shots (List[int]): 炮号列表
    name_table_columns (Dict): 表头字典
    t_min (float, optional): 起始时间
    t_max (float, optional): 结束时间
    resolution (float, optional): 分辨率

    返回:
    message: 操作成功或失败的消息
    """
    try:
        params = {
            "project_name": project_name,
            "shots": shots,
            "name_table_columns": name_table_columns,
        }
        if t_min is not None:
            params["t_min"] = t_min
        if t_max is not None:
            params["t_max"] = t_max
        if resolution is not None:
            params["resolution"] = resolution
        #if filter_list:
        #    params["filter_list"] = filter_list
        #print(filter_list)
        # export data
        response = requests.post(f"{FILE_SERVER_URL}/export/", json=params)
        response.raise_for_status()
        result = response.json()
        print(result)
        # add url to result
        if "urls" in result:
            result["urls"] = {shot: f"{FILE_SERVER_URL}{url}" for shot, url in zip(shots, result["urls"])}
        return result
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def export_imgs(
        project_name: str, 
        shots: List[int], 
        t_min: Optional[float|List[float]]=None, t_max: Optional[float|List[float]]=None, 
        resolution: Optional[float]=None
        ):
    """
    从高速摄影机的放电视频中导出图像

    参数:
    project_name (str): 项目名称, 也是文件夹名称
    shots (List[int]): 炮号列表
    t_min (float, optional): 起始时间
    t_max (float, optional): 结束时间
    resolution (float, optional): 时间分辨率

    返回:
    message: 操作成功或失败的消息
    """
    try:
        params = {
            "project_name": project_name,
            "shots": shots,
        }
        if t_min is not None:
            params["t_min"] = t_min
        if t_max is not None:
            params["t_max"] = t_max
        if resolution is not None:
            params["resolution"] = resolution
        # extract imgs from video
        response = requests.post(f"{EXTRACTOR_SERVER_URL}/extract/frames", json=params)
        response.raise_for_status()
        result = response.json()
        #add url to result
        img_urls = result.get('urls', {})
        if img_urls:
            result['urls'] = {
                shot: [FILE_SERVER_URL + url for url in url_list] if url_list else ""
                for shot, url_list in img_urls.items()
            }
        else:
            # if urls is {}
            result['urls'] = {}    
        return result
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def parse_urls_shots(urls:List[str]|str, data_type:str='timeseries'):
    '''
    从 url 中提取炮号信息

    参数：
    ----
    urls: List[str]|str
        单个放电数据链接或者一组链接
    data_type: str
        数据类型，timeseries或者image
    返回：
    ----
    data_dict: Dict[int, str]
        炮号和url对应的字典
    '''
    if isinstance(urls, str):
        urls = [urls]

    return {
        int(match.group(1)): url
        for url in urls
        if (match := re.search(r'/(\d+)\.csv$', url)) is not None
    }

def parse_urls_frame_times(urls: List[str]) -> List[float]:
    """
    从url中读取frame_time(帧时间)
    """
    TIME_PATTERN = re.compile(r"/(\d+)\.jpg$")

    def _parse_time_from_url(url: str) -> Optional[float]:
        m = TIME_PATTERN.search(url)
        return float(m.group(1)) / 1e5 if m else None
    
    times = [_parse_time_from_url(u) for u in urls]
    return times

def connect_redis(db: int=REDIS_DB):
    """
    连接 Redis

    参数:
    db (int, optional): Redis 数据库编号

    返回:
    redis.Redis: Redis 连接对象
    """
    redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=db)   
    return redis_conn

def export_urls_to_redis(
        project_name: str, 
        urls: Dict[int, Union[str, List[str]]],
        key: str="csv",
        db: int=REDIS_DB,
        frame_time: Dict[int, List[float]]= {}
        ):
    """
    导出数据到 Redis

    参数:
    project_name (str): 项目名称
    urls (Dict[int, Union[str, List[str]]]): 数据 URL 列表
    key (str, optional): 数据键名
    db (int, optional): Redis 数据库编号
    frame_time (Dict[int, List[float]]): 帧时间。仅用于保存图片数据至Redis

    return:
    message: 操作成功或失败的消息
    """
    try:
        # connect to redis
        redis_conn = connect_redis(db)
        # save urls to redis
        for shot, url in urls.items():
            if isinstance(url, str):
                data_dict ={key : url, "shot" : shot}
                redis_conn.set(f"{project_name}/{shot}", json.dumps({"data": data_dict}))
            elif isinstance(url, list):
                if frame_time.get(shot, None):
                    times = frame_time[shot]
                else:
                    times = []
                    for u in url:
                        m = re.search(r"/(\d+)\.jpg$", url)
                        times.append(float(m.group(1))/1e5 if m else "unknown")
                for u, t in zip(url, times):
                    data_dict = {key: u, "shot": shot, "frame_time": t}
                    suffix = int(round(t * 1e5))
                    redis_conn.set(f"{project_name}/{shot}_{suffix}",
                                   json.dumps({"data": data_dict}))
            else:
                redis_conn.close()
                return {"message": "unidentified type of variables"}
        # close redis connection
        redis_conn.close()
        return {
            "message": "success", 
            "config": {
                "host": REDIS_HOST,
                "port": REDIS_PORT,
                "db": db,
                "password": REDIS_PASSWORD
            }
        }
    except Exception as e:
        return {"message": str(e)}
    
def redis_paths():
    '''
    获取redis数据库中已有的path

    返回：
        path_counts: dict
        key是path，value是该路径下的文件数
    '''
    conn = connect_redis()
    
    # 获取所有键
    keys = conn.scan_iter('*')
    conn.close()

    path_counts = defaultdict(int)

    for key in keys:
        if '/' in key:
            top_path = key.split('/')[0]
            path_counts[top_path] += 1

    # 输出统计结果
    logger.info("一级路径总数：", len(path_counts))
    for a_path, count in path_counts.items():
        logger.info(f"{a_path} 下有 {count} 个文件（或子键）")
    return dict(path_counts) 

def list_redis_files(dir: str, redis_conn=None):
    '''
    获取指定路径下所有的文件的key

    参数：
    ----
    dir: str
        redis的路径
    redis_conn
        redis客户端

    返回：
    ----
    files: list
        redis指定路径的keys
    '''
    if redis_conn is None:
        redis_conn = connect_redis()
    pattern = f"{dir}/*"
    files = list(redis_conn.scan_iter(match=pattern))
    logger.info(f'{len(files)} 个匹配 key')
    redis_conn.close()
    return files


def load_data(urls: Dict[int, str], max_workers: int = 20) -> Dict[int, pd.DataFrame]:
    '''
    从多个 URL 并发加载数据

    参数:
    ----
    urls : Dict[int, str]
        键为 shot（炮号），值为对应 CSV 的 URL
    max_workers : int
        最大并发线程数（默认 20）

    返回:
    ----
    data_dict : Dict[int, pd.DataFrame]
        成功加载的数据，按 shot 编号存储
    '''
    def load_single_csv(shot_url: Tuple[int, str]) -> Tuple[int, pd.DataFrame]:
        """加载单个文件"""
        shot, url = shot_url
        try:
            df = pd.read_csv(url)
            logger.info(f"成功加载: {url}")
            return (shot, df)
        except Exception as e:
            logger.error(f"加载 {url} 时出错: {e}")
        return (shot, None)
    
    data_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_shot = {executor.submit(load_single_csv, item): item[0] for item in urls.items()}
        for future in as_completed(future_to_shot):
            shot = future_to_shot[future]
            try:
                shot_result, df = future.result()
                if df is not None:
                    data_dict[shot_result] = df
            except Exception as e:
                logger.error(f"线程执行 shot={shot} 出错: {e}")
    return data_dict
    
def _read_img_from_bytes(buf: bytes, as_gray: bool = False) -> np.ndarray:
    """把 bytes 转 OpenCV / numpy 格式."""
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    img_array = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(img_array, flag)
    if img is None:
        raise ValueError("cv2.imdecode 失败，可能文件损坏")
    return img

def load_imgs(urls: Dict[int, List[str]],
              max_workers: int = 20,
              as_gray: bool = False
             ) -> Dict[int, List[np.ndarray]]:
    """
    并发保序从 URL 列表中加载图像 

    参数
    ----
    urls: Dict[int, List[str]]
        键为 shot，值为该炮对应的一组图像 URL
    max_workers: int
        最大并发线程数
    as_gray: bool
        True 时返回灰度图 (H×W)，否则返回 BGR 彩色 (H×W×3)

    返回
    ----
    img_dict : Dict[int, List[np.ndarray]]
        每个 shot 对应一组 numpy 图像
    """
    session = requests.Session()
    def load_single_img(shot_id: int, idx: int, url: str) -> Tuple[int, np.ndarray]:
        try:
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
            img = _read_img_from_bytes(resp.content, as_gray)
            logger.info(f"成功加载: {url}")
            return shot_id, idx, img
        except Exception as e:
            logger.error(f"加载 {url} 时出错: {e}")
            return shot_id, idx, None
    
    img_dict: Dict[int, List[np.ndarray]] = {s: [None] * len(urls[s]) for s in urls}


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_single_img, s, i, u)
                   for s, lst in urls.items()
                   for i, u in enumerate(lst)]
        for future in as_completed(futures):
            try:
                shot, idx, img = future.result()
                img_dict[shot][idx] = img
            except Exception as e:
                logger.error(f"线程执行 shot={shot} 出错: {e}")

    # 删除为列表中为空的元素
    img_dict = {s: [im for im in lst if im is not None] for s, lst in img_dict.items()}
    return img_dict


def upload_dataframe(dir: str, file: str, dataframe: pd.DataFrame):
    """
    上传数据框到文件服务器

    参数:
    dir (str): 目录路径
    file (str): 文件名，建议以shot_id.csv的格式命名
    dataframe (pd.DataFrame): 数据框

    返回:
    message: 操作成功或失败的消息
    """
    try:
        # Convert DataFrame to in-memory CSV file
        csv_buffer = BytesIO()
        dataframe.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Reset buffer position to beginning
        # Upload directly
        file_path = f"{dir}/{file}"
        response = requests.post(
            f"{FILE_SERVER_URL}/upload/",
            files={"file": (file_path, csv_buffer, "text/csv")}
        ) 
        response.raise_for_status()
        # return url
        result = response.json()
        if "url" in result:
            result["url"] = f"{FILE_SERVER_URL}{result['url']}"
        return result
    except Exception as e:
        return {"error": str(e)}


def delete_redis_file(dir: str, file: Optional[str]=None, max_workers: int = 20, batch_size: int = 50):
    '''
    删除redis文件，支持多线程优化

    参数:
    dir (str): 目录路径
    file (str, optional): 文件名
    max_workers (int): 最大线程数
    batch_size (int): 每个线程处理的 key 数量
    '''
    def delete_keys(keys: List[str]):
        conn = connect_redis()
        if keys:
            conn.delete(*keys)
            logger.info(f"删除了 {len(keys)} 个 key")

    conn = connect_redis()
    if file is not None:
        key = f"{dir}/{file}"
        if conn.exists(key):
            conn.delete(key)
            logger.info(f'已删除文件 {key}')
        else:
            logger.info(f'文件 {key} 不存在')    
    else:
        pattern = f"{dir}/*"
        file_to_delete = list(conn.scan_iter(match=pattern))
        if file_to_delete:
            logger.info(f"共找到 {len(file_to_delete)} 个匹配 key，开始多线程删除")
            # 分批
            batches = [file_to_delete[i:i + batch_size] for i in range(0, len(file_to_delete), batch_size)]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(delete_keys, batches)
            logger.info(f"完成删除操作")
        else:
            logger.info("没有找到匹配的文件")


def shot_range(shot_min:int, shot_max:int):
    '''
    批量获取炮号shot
    '''
    conn = pg.connect_to_database()
    shots = pg.shot_range(conn=conn, shot_min=shot_min, shot_max=shot_max)
    pg.close_connection(conn)
    return shots


def load_dataframe(shot:int, name_table_columns:Dict, t_min = None, t_max=None, resolution = None):
    '''
    加载数据
    '''
    conn = pg.connect_to_database()
    kwargs = {
        "conn": conn,
        "shot": shot,
        "name_table_columns": name_table_columns
    }
    if t_min is not None:
        kwargs["t_min"] = t_min
    if t_max is not None:
        kwargs["t_max"] = t_max
    if resolution is not None:
        kwargs["resolution"] = resolution
    t, data = pg.load_data(**kwargs)
    pg.close_connection(conn)
    data_df = pd.DataFrame({"time": t})
    
    for name, array_data in data.items():
        # array_data.shape = (n_cols, len(t))，转置后成为 (len(t), n_cols)
        array_data_T = array_data.T
    
        # 获取对应的列名
        _, col_names = name_table_columns[name]
        if name == 'view_data':
            df_temp = pd.DataFrame(array_data_T, columns=[f"{col}" for col in col_names])
        else:    
            df_temp = pd.DataFrame(array_data_T, columns=[f"{name}_{col}" for col in col_names])
    
        # 拼接临时表与主表（按列）
        data_df = pd.concat([data_df, df_temp], axis=1)
    return data_df


def parse_expression(expr: str, raw_data_map: dict) -> str:
    """
    将配置文件中的表达式转换成PostgreSQL可识别的 SQL 表达式

    参数：
    - expr: str 
        配置文件中的字符串表达式
    - raw_data_map: dict 

    返回：
    - 转换后的 SQL 表达式字符串
    """

    # 用正则找到所有形如  name[channel] 的部分
    pattern = r'([a-zA-Z_]\w*)\[([\w]+)\]'  # group1: data名, group2: channel名

    def replace_channel(m):
        data_name = m.group(1)
        channel   = m.group(2)
        if data_name not in raw_data_map:
            raise ValueError(f"表达式中出现了未在 raw_data 中定义的数据名: {data_name}")

        table_name, channel_list = raw_data_map[data_name]
        if str(channel) not in map(str, channel_list):
            raise ValueError(f"表达式中 channel={channel} 不在 {data_name} 的 channels 列表中: {channel_list}")

        if str(channel).isdigit():
            return f'{table_name}.\"{channel}\"'
        else:
            return f'{table_name}.{channel}'

    # 用 re.sub 把形如  name[channel] 统一替换成 table."channel"
    sql_expr = re.sub(pattern, replace_channel, expr)
    return sql_expr

def delete_view(shots, view_name: str):
    '''
    删除指定scheme(shot)下的视图
    '''
    conn = pg.connect_to_database()
    cur = conn.cursor()
    if not isinstance(shots, list):
        shots = [shots]
    for shot in shots:
        try:
            delete_sql = f'DROP VIEW IF EXISTS "{shot}"."{view_name}" CASCADE'
            cur.execute(delete_sql)
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f'删除 {shot} 下的视图 {view_name} 失败，报错：{e}')
    conn.close()

def create_view(
    shots: List[int],
    raw_data_map: dict,
    processed_data: List[Dict],
    view_name: str = "feature_data_view",
    including_raw_data: bool =True,
    keep_used_raw_data: bool = False
) -> List :
    """
    传入一批 shot 和相关参数创造或替换VIEW

    参数:
    ----
    - shots: List[int] 炮号列表
    - raw_data_map: dict 原始数据构成
    - processed_data: dict 新数据的处理方式
    - view_name: str 一般为project name
    - including_raw_data: bool view是否包含原始数据，默认为False
    - keep_used_raw_data: bool 是否包含使用过的原始数据，默认为True

    返回：
    ----
    - view_columns: List 创建的 VIEW 的列名
    """

    def extract_and_convert(expression):
        pattern = r'([a-zA-Z0-9_]+)\[([a-zA-Z0-9_]+)\]'
        matches = re.findall(pattern, expression)
        converted = [f"{match[0]}_{match[1]}" for match in matches]
        return converted

    def generate_create_view_sql(
        raw_data_map: dict,
        processed_data: List[Dict],
        view_name: str = "feature_data_view",
        including_raw_data: bool = True,
        keep_used_raw_data: bool = False
    ):
        """
        生成一个「不含 shot/schema」的 SQL VIEW 创建脚本模版,其中 schema 用 {SCHEMA} 作占位符

        返回
        ----
        create_view_sql_template: str 
            SQL 字符串, 会包含:
            CREATE OR REPLACE VIEW {SCHEMA}."view_name" AS
            SELECT ...
            FROM {SCHEMA}.t
            LEFT JOIN {SCHEMA}.xxx ...
            ...
        column_list: List
            VIEW中会包含的列
        """

        select_parts = ['t.id AS t_id']  # 加入t_id便于多表查询，可以兼容postgre现有的导出数据的接口
        join_parts = []
        # 这里 schema 用占位符 {SCHEMA}
        join_parts.append(f"FROM {{SCHEMA}}.t")
        column_list = []
        used_columns = set()

        for item in processed_data:
            new_col_name = item["name"]          
            expr = item["expression"]
            used_columns.update(extract_and_convert(expr))          
            sql_expr = parse_expression(expr, raw_data_map)
            select_parts.append(f"({sql_expr})::real AS \"{new_col_name}\"")
            column_list.append(new_col_name)

        # 遍历 raw_data_map, 生成 JOIN
        for data_alias, (table_name, channel_list) in raw_data_map.items():
            table_alias = table_name.strip('"')
            join_parts.append(
                f"INNER JOIN {{SCHEMA}}.{table_name} AS {table_alias} ON (t.id = {table_alias}.t_id)"
            )

            if including_raw_data:
                for ch in channel_list:
                    col_alias = f"{data_alias}_{ch}"
                    if keep_used_raw_data and col_alias in used_columns:
                        continue
                    select_parts.append(f"{table_alias}.{ch} AS \"{col_alias}\"")
                    column_list.append(col_alias)


        select_sql = ",\n       ".join(select_parts)
        join_sql   = "\n".join(join_parts)

        create_view_sql_template = f"""
        CREATE OR REPLACE VIEW {{SCHEMA}}."{view_name}" AS
        SELECT
        {select_sql}
        {join_sql}
        ORDER BY t.id;
        """
        return create_view_sql_template.strip(), column_list

    sql_template, view_columns = generate_create_view_sql(raw_data_map, processed_data, view_name, including_raw_data ,keep_used_raw_data)
    conn = connect_to_database()
    for shot in shots:
        # 将 shot 转换为 {SCHEMA} = "shot"
        schema_str = f"\"{shot}\""
        create_view_sql = sql_template.replace("{SCHEMA}", schema_str)
        execute_query(conn, create_view_sql)
        logger.info(f"[INFO] CREATE VIEW: schema={shot}")
    close_connection(conn)
    return view_columns


def filter_data(
        urls: Dict[int, str],
        filters: List[Callable[[pd.DataFrame], bool]] = [],
        delete_filtered_data: bool = True,
        dir: str = None,
        max_workers: int = 20
    ):
    """
    并发加载数据并进行过滤，可返回被筛掉的数据，并支持删除。

    参数:
    ----
    urls : Dict[int, str]
        炮号 -> 文件 URL
    filters : List[Callable]
        过滤器列表，对每个 DataFrame 判断是否保留。为空表示不过滤。
    delete_filtered_data : bool
        是否删除不满足条件的数据（包括文件和Redis）
    dir : str
        数据所在目录，用于删除操作
    max_workers : int
        并发线程数

    返回:
    ----
        urls : Dict[int, str]  通过过滤的数据URL
        若filter不为空，还会返回未通过数据的炮号
        filtered_out : List[int]  未通过的数据的炮号
    ]
    """
    if delete_filtered_data and dir is None:
        raise ValueError("dir should not be None when delete_filtered_data is True")

    def load_single_csv(shot_url: Tuple[int, str]) -> Tuple[int, pd.DataFrame]:
        shot, url = shot_url
        try:
            df = pd.read_csv(url)
            return (shot, df)
        except Exception as e:
            logger.error(f"加载 {shot}, {url} 时出错: {e}")
            return (shot, None)

    filtered_out = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_shot = {executor.submit(load_single_csv, item): item[0] for item in urls.items()}
        for future in as_completed(future_to_shot):
            shot = future_to_shot[future]
            try:
                shot_result, df = future.result()
                if df is not None:
                    # 不进行过滤，或者全部过滤通过
                    if filters and (not all(f(df) for f in filters)):
                        logger.info(f"数据不符合要求: shot {shot_result}")
                        del urls[shot_result]
                        filtered_out.append(shot_result)
                        if delete_filtered_data:
                            delete_file(dir, f"{shot_result}.csv")
                            delete_redis_file(dir, f"{shot_result}")
            except Exception as e:
                logger.error(f"线程执行 shot={shot} 出错: {e}")
        return urls, filtered_out 