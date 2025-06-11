# 复用你的 connect_annotation_database()
import psycopg2, json, datetime
from psycopg2 import sql
from psycopg2.extras import Json, execute_values
from typing import List, Dict, Any, Sequence, Optional
from sunist2.script.camera import fps, num_frames, trig_time, height, width
import numpy as np
from .utils import connect_annotation_database, delete_shot_schema, delete_table, get_table_columns, list_existing_tables 

MICRO_FACTOR = 1e5

# ---------- 建表 ----------
def create_shot_schema(
        pg_conn: psycopg2.extensions.connection, 
        shot: int
        )-> dict:
    '''
    根据炮号创建schema并创建frames主表

    参数：
    pg_conn: pgsql客户端
    shot: int 炮号

    返回：
    frames表的frame_time和对应的id
    '''
    schema = f'"{shot}"'
    cur = pg_conn.cursor()

    # 判断 schema 和 frames 表是否存在
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_schema = %s AND table_name = 'frames'
        );
    """, (f'{shot}',))
    exists = cur.fetchone()[0]
    if exists:
        print(f"Schema '{shot}' with table 'frames' already exists. Skipping creation.")
        cur.close()
        return {}# 直接返回

    cur.execute(f'CREATE SCHEMA IF NOT EXISTS {schema};') # 创建schema
    # 创建帧时间主表
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {schema}.frames (
        id SERIAL PRIMARY KEY,
        frame_time BIGINT UNIQUE,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """) 
    pg_conn.commit()
    # 根据实际视频参数插入主表
    trig = int(trig_time(shot) * (MICRO_FACTOR / 1e3))
    del_time = int(MICRO_FACTOR/fps(shot))
    num = num_frames(shot)
    frame_time = np.arange(trig, trig+del_time*num, del_time).tolist()

    sql = f"""
      INSERT INTO {schema}.frames (frame_time)
      VALUES %s
      RETURNING id, frame_time;
    """
    res = execute_values(cur, sql, [(ft,) for ft in frame_time], fetch=True)
    # [(id, frame_time), ...]
    pg_conn.commit()
    cur.close()
    return {float(ft)/MICRO_FACTOR: fid for fid, ft in res}   

def create_image_table(
        pg_conn: psycopg2.extensions.connection,
        shot:int,
        table_name: str,
        label_columns: Sequence[str]= None,
        unique_image: bool = False,
    ):
    """
    创建表(如果不存在).
        pg_conn：Postgresql客户端
        shot:int schema名
        table_name: 表名
        label_columns: 自定义列，主要用于包括label类型等，默认包括"label"列
        unique_image: 是否为(frame_id)加 unique约束(意味着一张图片只能有一组mask)
    """
    # 构造列
    schema = f'"{shot}"'
    table = f'"{table_name}"'
    label_columns = label_columns or ["label"]
    label_columns = [f'"{column}"' for column in label_columns]
    cols = [
        "id SERIAL PRIMARY KEY",
        f"frame_id INT REFERENCES {schema}.frames(id) ON DELETE CASCADE",
        "geom_type VARCHAR(20)",
        "geom_data JSONB",
        "annotator INT",
        "annotation_id INT",
        "created_at TIMESTAMPTZ DEFAULT NOW()",
        "updated_at TIMESTAMPTZ DEFAULT NOW()",
        "annotation_created TIMESTAMPTZ DEFAULT NOW()",
        "annotation_updated TIMESTAMPTZ DEFAULT NOW()",
    ]
    cols.extend([f"{column} VARCHAR(20)" for column in label_columns])
    constraint_sql = '''CONSTRAINT geom_type_geom_data_coherence CHECK (
        (geom_type IS NULL  AND geom_data IS NULL)  OR
        (geom_type IS NOT NULL AND geom_data IS NOT NULL))'''
    col_def = ",\n   ".join(cols)
    sql = f"CREATE TABLE IF NOT EXISTS {schema}.{table} (\n   {col_def},\n{constraint_sql}\n);"
    print(sql)
    cur = pg_conn.cursor()
    cur.execute(sql)
    pg_conn.commit()

    if unique_image:
        # 为(shot)加唯一约束
        alter_sql = f"""ALTER TABLE {schema}.{table} 
                        ADD CONSTRAINT {table}_unique_image UNIQUE(frame_id);"""

    else:
        alter_sql = f"""CREATE INDEX IF NOT EXISTS idx_{table_name}_frame_id
                        ON {schema}.{table} (frame_id);""" 
    try:
            print(alter_sql)
            cur.execute(alter_sql)
            pg_conn.commit()
    except psycopg2.errors.DuplicateTable:
        # 说明已经加过 unique 约束, 可以忽略或 pass
        pg_conn.rollback()    
                       
    cur.close()

def create_number_table( 
        pg_conn: psycopg2.extensions.connection,
        shot:int,
        table_name: str,
        num_columns: Sequence[str],
        label_columns: Sequence[str]=None,
        unique_image: bool = False,
    ):
    """
    创建表(如果不存在).
        pg_conn：Postgresql客户端
        shot:int schema名
        table_name: 表名
        num_columns: 自定义列，用于存储参数数据。请保证和将要插入的数据相对应。
        label_columns: 自定义列，主要用于包括label类型等，默认包括"label"列
        unique_image: 是否为(frame_id)加 unique约束(意味着一张图片只能有一组mask)
    """
    # 构造列
    schema = f'"{shot}"'
    table = f'"{table_name}"'
    #label_columns = label_columns or ["label"]
    label_columns = [f'"{column}"' for column in label_columns]
    num_columns = [f'"{column}"' for column in num_columns]
    cols = [
        "id SERIAL PRIMARY KEY",
        f"frame_id INT REFERENCES {schema}.frames(id) ON DELETE CASCADE",
        "annotator INT",
        "annotation_id INT",
        "created_at TIMESTAMPTZ DEFAULT NOW()",
        "updated_at TIMESTAMPTZ DEFAULT NOW()",
        "annotation_created TIMESTAMPTZ DEFAULT NOW()",
        "annotation_updated TIMESTAMPTZ DEFAULT NOW()",
    ]
    cols.extend([f"{column} VARCHAR(20)" for column in label_columns])
    cols.extend([f"{column} DOUBLE PRECISION" for column in num_columns])
    col_def = ",\n   ".join(cols)
    sql = f"CREATE TABLE IF NOT EXISTS {schema}.{table} (\n   {col_def}\n);"
    print(sql)
    cur = pg_conn.cursor()
    cur.execute(sql)
    pg_conn.commit()

    if unique_image:
        # 为(shot)加唯一约束
        alter_sql = f"""ALTER TABLE {schema}.{table} 
                        ADD CONSTRAINT {table_name}_unique_image UNIQUE(frame_id);"""

    else:
        alter_sql = f"""CREATE INDEX IF NOT EXISTS idx_{table_name}_frame_id
                        ON {schema}.{table} (frame_id);""" 
    try:
            print(alter_sql)
            cur.execute(alter_sql)
            pg_conn.commit()
    except psycopg2.errors.DuplicateTable:
        # 说明已经加过 unique 约束, 可以忽略或 pass
        pg_conn.rollback()

    cur.close()    

# ---------- 插入标注 ----------

def fetch_frame_time(
        pg_conn, 
        shot:int|str, 
        convert_to_second:bool=True
    ):
    """读取 {frame_time: id} 映射"""
    schema_ident = sql.Identifier(str(shot))
    with pg_conn.cursor() as cur:
        cur.execute(sql.SQL("""
            SELECT frame_time, id
            FROM {}.frames
            ORDER BY frame_time
        """).format(schema_ident))
        rows = cur.fetchall()       # [(frame_time, id), ...]
    return {float(ft)/MICRO_FACTOR: fid for ft, fid in rows} if convert_to_second else {ft: fid for ft, fid in rows}

def insert_annotations(
        pg_conn, 
        shot:int,
        table_name:str,
        ann_list: List[Dict[str, Any]],
        on_conflict :tuple|str=None):
    """
    ann_list: [{
        'frame_time':0.054,
        'geom_type':'mask',
        'geom_data': {'rle':'1 3 10 2'},
        'labels':['Edge'],
        'annotator':'alice',
        ...
    }]
    """
    schema = f'"{shot}"'

    all_keys = {k for ann in ann_list for k in ann} - {'shot', 'image_height', 'image_width'}
    # 强制包含 frame_time
    if "frame_time" not in all_keys:
        raise ValueError("Each annotation must have at least 'frame_time'")
    all_keys.discard('frame_time')
    
    # 检查数据和表是否匹配
    columns = set(get_table_columns(pg_conn, table_name, schema=f'{shot}'))
    if not all_keys.issubset(columns):
        raise ValueError(f"data keys don't match table columns: {all_keys - columns}")
    col_list = sorted(all_keys)
    quoted_cols = [f'"{c}"' for c in col_list]
    
    # on conflict
    conflict_clause = ""
    if on_conflict:
    # 把列提取出来逐个加引号，兼容传 "(frame_id,R)" 或 "frame_id"
        if on_conflict.startswith("(") and on_conflict.endswith(")"):
            conflict_cols = [c.strip() for c in on_conflict[1:-1].split(",")]
        else:
            conflict_cols = [on_conflict.strip()]

        quoted_conflict = "(" + ",".join(f'"{c}"' for c in conflict_cols) + ")"
        sets = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in col_list)
        conflict_clause = f"ON CONFLICT {quoted_conflict} DO UPDATE SET {sets}"

    # 获取frame_time和time的映射关系
    frame_id_map = fetch_frame_time(pg_conn, shot, False)
    # 组装行数据
    rows_values = []
    for ann in ann_list:
        #d['annotation_created'] = self.string_to_timestampz(d.get('annotation_created',None))
        #d['annotation_updated'] = self.string_to_timestampz(d.get('annotation_updated',None))
        row = []
        for col in col_list:
            # 保证没有则 None
            data = ann.get(col, None)
            if isinstance(data, dict):
                data = Json(data)
            row.append(data)
        row.append(frame_id_map[int(round(MICRO_FACTOR * ann['frame_time']))])    
        rows_values.append(row)
    col_list.append('frame_id')
    quoted_cols.append('"frame_id"')
    
    sql = f"""
      INSERT INTO {schema}.{table_name} ({", ".join(quoted_cols)})
      VALUES %s
      {conflict_clause}
      RETURNING id;
    """
    print(sql)
    with pg_conn.cursor() as cur:
        execute_values(cur, sql, rows_values,fetch=True)
        pg_conn.commit()


# --------- 查询数据 ---------
def query_image_annotations(
    pg_conn,
    shots: Sequence[int]|int,
    table_name: str,
    start_time: Optional[float] = None,   
    end_time:   Optional[float] = None,   
    filters: Optional[Dict[str, Any]] = None,
    columns: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    根据 shot／时间窗口／附加列筛选条件导出图像标注
    
    参数：
    shots      : 要查询的炮号列表或单独一炮
    table_name : 表名，必须
    start_time : 起始时间（秒）。为 None 则不限
    end_time   : 结束时间（秒）。为 None 则不限
    filters    : 其它列的筛选，如 {"label": "Edge", "geom_type": "mask"}
    columns    : 想返回的列，默认为 ['frame_time','geom_type','geom_data',
                                    'label','annotator','annotation_id']

    返回：List[dict]
    图像标注的查询结果
    """
    if not shots:
        return []
    if isinstance(shots, int):
        shots = [shots]

    # 默认返回列
    default_cols = [
        "frame_time", "geom_type", "geom_data",
        "annotator", "annotation_id"
    ]
    columns = list(dict.fromkeys(columns or default_cols))  # 去重保持顺序
    if "frame_time" not in columns:
        columns.insert(0, "frame_time")

    results: List[Dict[str, Any]] = []
    filters = filters or {}

    for shot in shots:
        schema_ident = sql.Identifier(str(shot))
        table_ident  = sql.Identifier(table_name)

        # ---------- SELECT 列 ----------
        select_parts = [sql.SQL("fr.frame_time")]
        for c in columns:
            if c == "frame_time":
                continue
            select_parts.append(sql.SQL("ai.{}").format(sql.Identifier(c)))
        select_sql = sql.SQL(", ").join(select_parts)

        # ---------- 动态 WHERE ----------
        where_parts: List[sql.Composed] = []
        params: List[Any] = []

        # 时间过滤
        if start_time is not None:
            where_parts.append(sql.SQL("fr.frame_time >= %s"))
            params.append(int(round(start_time * MICRO_FACTOR)))
        if end_time is not None:
            where_parts.append(sql.SQL("fr.frame_time <= %s"))
            params.append(int(round(end_time * MICRO_FACTOR)))

        # 其它列过滤
        for col, val in filters.items():
            where_parts.append(sql.SQL("ai.{} = %s").format(sql.Identifier(col)))
            params.append(val)

        where_sql = (
            sql.SQL("WHERE ") + sql.SQL(" AND ").join(where_parts)
            if where_parts else sql.SQL("")
        )

        query = sql.SQL("""
            SELECT {fields}
            FROM {schema}.{table} AS ai
            JOIN {schema}.frames AS fr
              ON fr.id = ai.frame_id
            {where}
            ORDER BY fr.frame_time
        """).format(
            fields = select_sql,
            schema = schema_ident,
            table  = table_ident,
            where  = where_sql
        )

        try:
            with pg_conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                if not rows:
                    continue
        except Exception as e:
            pg_conn.rollback()   
            print(f'error:{e}')     

            # 列名映射
            col_names = ["frame_time"] + [c for c in columns if c != "frame_time"]
            for r in rows:
                res = {}
                for i, col in enumerate(col_names):
                    res[col] = (float(r[i]) / MICRO_FACTOR) if col == "frame_time" else r[i]
                res["shot"] = shot
                results.append(res)

    return results


def query_number_table(
    pg_conn,
    shot: int,
    table_name: str,
    param_columns: Optional[Sequence[str]] = None,    # None => 自动识别
    start_time: Optional[float] = None,               # 秒
    end_time: Optional[float] = None,
    as_array: bool = False,                           # True => (t, ndarray)
    interpolate: bool = False,
    target_resolution: Optional[float] = None,
    extra_filters: Optional[Dict[str, Any]] = None
):
    """
    取指定炮号的 number 参数（可选直接线性插值）。

    返回
    ----
    - as_array=False:  List[Dict]  (frame_time, p1, p2, ...)
    - as_array=True :  (t   : np.ndarray shape (N,),
                        data: np.ndarray shape (N, n_params),
                        cols: List[str])              # 参数列顺序

    Notes
    -----
    * interpolate=True 时必须提供 target_resolution
    * start_time/end_time 单位为 **秒**；
    """

    NUMERIC_TYPES = {
        "double precision", "real", "numeric",
        "decimal", "float"
        }


    schema = sql.Identifier(str(shot))
    tbl    = sql.Identifier(table_name)
    extra_filters = extra_filters or {}

    # -------- 1. 解析参数列 --------
    if param_columns is None:
        param_columns = get_table_columns(
            pg_conn, table_name, schema=str(shot),
            include_type=NUMERIC_TYPES,
            exclude_columns=("id", "frame_id","annotation_id","annotator"),
            name_only=True                # 常见非参数列
        )
        if not param_columns:
            raise ValueError("自动识别不到任何数值列，请手动传入 param_columns。")

    # -------- 2. 构造 SQL --------
    sel_cols_sql = [
        sql.SQL("fr.frame_time")
    ] + [
        sql.SQL("np.{}").format(sql.Identifier(c)) for c in param_columns
    ]

    where_parts, params = [], []
    if start_time is not None:
        where_parts.append(sql.SQL("fr.frame_time >= %s"))
        params.append(int(round(start_time * MICRO_FACTOR)))
    if end_time is not None:
        where_parts.append(sql.SQL("fr.frame_time <= %s"))
        params.append(int(round(end_time * MICRO_FACTOR)))

    # 附加过滤
    for col, v in extra_filters.items():
        where_parts.append(sql.SQL("np.{} = %s").format(sql.Identifier(col)))
        params.append(v)

    where_sql = sql.SQL("WHERE ") + sql.SQL(" AND ").join(where_parts) if where_parts else sql.SQL("")

    q = sql.SQL("""
        SELECT {fields}
        FROM   {schema}.{tbl} AS np
        JOIN   {schema}.frames AS fr ON fr.id = np.frame_id
        {where}
        ORDER  BY fr.frame_time;
    """).format(
        fields = sql.SQL(", ").join(sel_cols_sql),
        schema = schema,
        tbl    = tbl,
        where  = where_sql
    )

    # -------- 3. 执行查询 --------
    with pg_conn.cursor() as cur:
        cur.execute(q, params)
        rows = cur.fetchall()             # [(ft, p1, p2, ...)]

    if not rows:                          # 空结果
        return (np.array([]), np.empty((0, len(param_columns))), param_columns) if as_array else []

    # -------- 4. 转为 np.array --------
    times = np.array([r[0] for r in rows], dtype=np.float64) / MICRO_FACTOR   # → 秒
    data  = np.array([r[1:] for r in rows], dtype=np.float64)          # shape (N, k)

    # -------- 5. 可选插值 --------
    if interpolate:
        if target_resolution is None:
            raise ValueError("interpolate=True 时必须指定 target_fps")
        t_uniform, data_uniform = _linear_interp(times, data, target_resolution)
        times, data = t_uniform, data_uniform

    if as_array:
        return times, data, list(param_columns)

    # 默认字典列表
    out = []
    for i, t in enumerate(times):
        d = {"shot": shot, "frame_time": float(t)}
        for j, col in enumerate(param_columns):
            d[col] = float(data[i, j])
        out.append(d)
    return out

def _linear_interp(
    t: np.ndarray,          
    y: np.ndarray,          
    resolution: float              
):
    """
    把 (t, y) 补齐到均匀时间轴，线性插值。
    返回 (t_uniform, y_uniform)
    """
    if t.ndim != 1 or y.ndim != 2:
        raise ValueError("t must be 1-D, y must be 2-D")
    if len(t) != len(y):
        raise ValueError("t, y length mismatch")

    t_uniform = np.arange(t[0], t[-1] + 0.5 * resolution, resolution)
    y_uniform = np.empty((len(t_uniform), y.shape[1]), dtype=y.dtype)

    for col in range(y.shape[1]):
        y_uniform[:, col] = np.interp(t_uniform, t, y[:, col])

    return t_uniform, y_uniform

        