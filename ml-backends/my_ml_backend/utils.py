from toklabel.config import FILE_SERVER_URL, REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB
import requests
from typing import Dict, List, Optional, Tuple
import redis
import json
from io import BytesIO
import pandas as pd
import os
import sunist2.script.postgres as pg
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

def list_files(dir: str):
    """
    列出目录下的所有文件

    参数:
    dir (str): 目录路径

    返回:
    List[str]: 列表，每个元素为文件的 URL
    """
    try:
        params = {"dir": dir} 
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
        # export data
        response = requests.post(f"{FILE_SERVER_URL}/export/", json=params)
        response.raise_for_status()
        result = response.json()
        # add url to result
        if "urls" in result:
            result["urls"] = {shot: f"{FILE_SERVER_URL}{url}" for shot, url in zip(shots, result["urls"])}
        return result
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


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
        urls: Dict[int, str],
        key: str="csv",
        db: int=REDIS_DB
        ):
    """
    导出数据到 Redis

    参数:
    project_name (str): 项目名称
    urls (List[str]): 数据 URL 列表
    key (str, optional): 数据键名
    db (int, optional): Redis 数据库编号

    return:
    message: 操作成功或失败的消息
    """
    try:
        # connect to redis
        redis_conn = connect_redis(db)
        # save urls to redis
        for shot, url in urls.items():
            data_dict ={key : url, "shot" : shot}
            redis_conn.set(f"{project_name}/{shot}", json.dumps({"data": data_dict}))
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

    path_counts = defaultdict(int)

    for key in keys:
        if '/' in key:
            top_path = key.split('/')[0]
            path_counts[top_path] += 1

    # 输出统计结果
    print("一级路径（A）总数：", len(path_counts))
    for a_path, count in path_counts.items():
        print(f"{a_path} 下有 {count} 个文件（或子键）")
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
    print(f'{len(files)} 个匹配 key')
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
            print(f"成功加载: {url}")
            return (shot, df)
        except Exception as e:
            print(f"加载 {url} 时出错: {e}")
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
                print(f"线程执行 shot={shot} 出错: {e}")
    return data_dict
    

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
        file_path = os.path.join(dir, file)
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
            print(f"删除了 {len(keys)} 个 key")

    conn = connect_redis()
    if file is not None:
        key = f"{dir}/{file}"
        if conn.exists(key):
            conn.delete(key)
            print(f'已删除文件 {key}')
        else:
            print(f'文件 {key} 不存在')    
    else:
        pattern = f"{dir}/*"
        file_to_delete = list(conn.scan_iter(match=pattern))
        if file_to_delete:
            print(f"共找到 {len(file_to_delete)} 个匹配 key，开始多线程删除")
            # 分批
            batches = [file_to_delete[i:i + batch_size] for i in range(0, len(file_to_delete), batch_size)]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(delete_keys, batches)
            print(f"完成删除操作")
        else:
            print("没有找到匹配的文件")


def shot_range(shot_min:int, shot_max:int):
    '''
    批量获取炮号shot
    '''
    conn = pg.connect_to_database()
    shots = pg.connect_to_database(conn=conn, shot_min=shot_min, shot_max=shot_max)
    return shots


def load_dataframe(shot:int, name_table_columns:Dict, t_min = None, t_max=None, resolution = None):
    '''
    加载数据
    '''
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
    conn = pg.connect_to_database()
    t, data = pg.load_data(**kwargs)
    pg.close_connection(conn)
    data_df = pd.DataFrame({"time": t})
    
    for name, array_data in data.items():
        # array_data.shape = (n_cols, len(t))，转置后成为 (len(t), n_cols)
        array_data_T = array_data.T
    
        # 获取对应的列名
        _, col_names = name_table_columns[name]
        df_temp = pd.DataFrame(array_data_T, columns=[f"{name}_{col}" for col in col_names])
    
        # 拼接临时表与主表（按列）
        data_df = pd.concat([data_df, df_temp], axis=1)
    return data_df

