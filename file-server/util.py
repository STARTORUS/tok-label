import pandas as pd
import sunist2.script.postgres as pg
from typing import Dict, List, Tuple    
import numpy as np


def export_postgres_data(
        shots: int|List[int], 
        name_table_columns: Dict[str, Tuple[str, List[str]]], 
        t_min: float=-np.inf, 
        t_max: float=np.inf, 
        resolution: float=1e-3
        ):
    """
    从PostgreSQL中读取指定炮号的数据，并将其转换为dict格式。
    
    参数:
    ----
    shot: int, str 或 可迭代对象
        炮号(可为数字或字符串)，也可以是包含多个炮号的可迭代对象。
    name_table_columns: dict
        形如:
        {
            "some_name": ("table_name", ["col1", "col2", ...]),
            "other_name": ("other_table", ["colA", "colB", ...])
        }
        字典的key随意，但value中需要指定数据库的表名和对应列名。
    t_min: float, 可选
        数据的最小时间，与 load_data 函数对应。
    t_max: float, 可选
        数据的最大时间，与 load_data 函数对应。
    resolution: float, 可选
        数据降采样分辨率(秒)，与 load_data 函数对应。
    
    返回:
    ----
    pd_data_dict
        返回一个字典，每个键为炮号，每个值为该炮号对应的数据DataFrame。
    """
    # 1. 连接数据库
    pd_data_dict = {}
    conn = pg.connect_to_database()
    try:
        # 将shot包装为列表
        if isinstance(shots, (str, int)):
            shots = [shots]
        elif hasattr(shots, '__iter__'):
            shots = list(shots)
        else:
            shots = [shots]
            
        # 针对每个shot处理数据
        for s in shots:
            t, data_dict = pg.load_data(
                conn,
                shot=s,
                name_table_columns=name_table_columns,
                t_min=t_min,
                t_max=t_max,
                resolution=resolution)
            # 建立一个初始DataFrame，将时间t设为第一列
            df_all = pd.DataFrame({"time": t})
    
            # 循环每个 key，将其数据合并到df_all
            for name, array_data in data_dict.items():
                # array_data.shape = (n_cols, len(t))，转置后成为 (len(t), n_cols)
                array_data_T = array_data.T
    
                # 获取对应的列名
                _, col_names = name_table_columns[name]
                if name == 'view_data':
                    df_temp = pd.DataFrame(array_data_T, columns=[f"{col}" for col in col_names])
                else:    
                    df_temp = pd.DataFrame(array_data_T, columns=[f"{name}_{col}" for col in col_names])
    
                # 拼接临时表与主表（按列）
                df_all = pd.concat([df_all, df_temp], axis=1)
            
            # 以shot作为key标记当前数据
            pd_data_dict[s] = df_all
    finally:
        # 关闭数据库连接
        pg.close_connection(conn)
    
    return pd_data_dict