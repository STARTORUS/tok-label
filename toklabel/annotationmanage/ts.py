import psycopg2
import psycopg2.sql as sql
from psycopg2.extras import execute_values
from ..config import POSTGRE_HOST, POSTGRE_PORT, POSTGRE_USER, POSTGRE_PASSWORD, POSTGRE_DATABASE
from sunist2.script.postgres import execute_query, query_data
from typing import Optional, Dict, Any, List, Sequence, Tuple
from datetime import datetime
import pytz
from .utils import connect_annotation_database, delete_shot_schema, delete_table, get_table_columns, list_existing_tables, string_to_timestampz


def create_annotation_table(
        pg_conn: psycopg2.extensions.connection,
        table_name: str,
        multi_label_group: bool = False,
        with_label: bool = True,
        label_name: str = None,
        unique_shot: bool = False,
        point_allowed: bool = True
    ):
    """
    创建表(如果不存在).
        pg_conn：Postgresql客户端
        table_name: 表名
        multi_label_group: 是否涉及多个标签组，若为True，表将增加label和feature这两个表头
        with_label: 是否包含 label字段
        label_name: label列的名称，仅在 multi_label_group=False 且 with_label 为 True 时起作用
        unique_shot: 是否为(shot)加 unique约束(意味着每个shot只能有一条记录)
        point_allowed: 是否允许存 is_point(可以不允许的话, 也可省略该列)
    """
    # 构造列
    cols = [
        "id SERIAL PRIMARY KEY",
        "shot INT NOT NULL",
    ]
    if multi_label_group:
        cols.append('feature VARCHAR(50)')
        cols.append('label VARCHAR(50)')        
    elif with_label:
        if label_name is None:
            label_name='label'
        cols.append(f"{label_name} VARCHAR(50)")
    cols.append("start_time DOUBLE PRECISION")
    cols.append("end_time DOUBLE PRECISION")
    if point_allowed:
        cols.append("is_point BOOLEAN DEFAULT FALSE")
    cols.append("annotator INT")
    cols.append("annotation_id INT")
    cols.append("annotation_created TIMESTAMPTZ DEFAULT NOW()")
    cols.append("annotation_updated TIMESTAMPTZ DEFAULT NOW()")
    cols.append("created_at TIMESTAMPTZ DEFAULT NOW()")
    cols.append("updated_at TIMESTAMPTZ DEFAULT NOW()")

    col_def = ",\n   ".join(cols)
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n   {col_def}\n);"
    cur = pg_conn.cursor()
    cur.execute(sql)
    pg_conn.commit()

    if unique_shot:
        # 为(shot)加唯一约束
        alter_sql = f"""ALTER TABLE {table_name} 
                        ADD CONSTRAINT {table_name}_unique_shot UNIQUE(shot);"""
        try:
            cur.execute(alter_sql)
            pg_conn.commit()
        except psycopg2.errors.DuplicateTable:
            # 说明已经加过 unique 约束, 可以忽略或 pass
            pg_conn.rollback()
    cur.close()        

def insert_annotations(
    pg_conn: psycopg2.extensions.connection,
    table_name: str,
    data_list: List[Dict[str, Any]],
    on_conflict: Optional[str] = None
):
    """
    插入若干条标注数据:
        data_list 里的每个dict应包含:
        shot: int
        start_time: float
        end_time: float
        (可选) label: str
        (可选) is_point: bool
        (可选) annotator: str
        (可选) annotation_id: int
        (可选) annotation_created: str
        (可选) annotation_updated: str
        (可选) annotation_id: int
        on_conflict: 若不为None, 形如 '(shot)' or '(shot, label)', 
                    则执行 on conflict do update.
    """
    if not data_list:
        return
    
    # 构造插入列
    # 取并集
    all_keys = set()
    for d in data_list:
        all_keys.update(d.keys())
    # 强制包含 shot, start_time, end_time
    if "shot" not in all_keys or "start_time" not in all_keys or "end_time" not in all_keys:
        raise ValueError("Each annotation must have at least 'shot','start_time','end_time'")
    # 检查数据和表是否匹配
    columns = set(get_table_columns(pg_conn, table_name))
    if not all_keys.issubset(columns):
        print(f'列名：{columns}')
        print(f'插入数据：{all_keys}')
        raise ValueError("data keys don't match table columns") 

    col_list = sorted(all_keys)  # e.g. ["annotation_id","annotator","end_time","is_point","label","shot","start_time"]
    print(col_list)
    # 构造参数列表
    rows_values = []
    for d in data_list:
        #d['annotation_created'] = self.string_to_timestampz(d.get('annotation_created',None))
        #d['annotation_updated'] = self.string_to_timestampz(d.get('annotation_updated',None))
        row = []
        for col in col_list:
            # 保证没有则 None
            row.append(d.get(col, None))
        rows_values.append(row)

    # on conflict
    conflict_clause = ""
    if on_conflict:
        if not on_conflict.startswith("("):
            on_conflict = f"({on_conflict})"
        conflict_clause = f"ON CONFLICT {on_conflict} DO UPDATE SET "
        # 更新全部字段(除主键). 
        update_parts = []
        for c in col_list:
            if c == "id":
                continue
            update_parts.append(f"{c} = EXCLUDED.{c}")
        conflict_clause += ", ".join(update_parts)

    # 构造 insert 语句
    col_str = ", ".join(col_list)
    insert_sql = f"""
        INSERT INTO {table_name} ({col_str})
        VALUES %s
        {conflict_clause}
    """
    print(insert_sql)
    cur = pg_conn.cursor()
    execute_values(cur, insert_sql, rows_values)
    pg_conn.commit()
    cur.close()

def get_shots_single_table(
    pg_conn: psycopg2.extensions.connection,
    table_name: str,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    feature_name: Optional[str] = None,
    label_name: Optional[str] = None,
    label_column_name: Optional[str] = 'label',
    has_feature: bool = True,
    has_label: bool = True
) -> List[int]:
    """
    根据条件从指定表中筛选符合条件的唯一炮号，并按升序返回
    
    参数:
    ----
    table_name: str 需要查询的表名
    min_duration: float 允许的最小持续时间(end_time - start_time)
    max_duration: float 允许的最大持续时间
    feature_name: str 指定特征名称（仅当表为多标签组时有效）
    label_name: str 指定标签名称
    label_column_name: str label列的表头名，默认为"label"
    has_feature: bool 是否要求存在特征（True时要求存在，False时要求不存在）
    has_label: bool 是否要求存在标签（True时要求存在，False时要求不存在）
    
    返回:
    ----
    shots: List[int] 排序后的唯一炮号列表
    
    示例:
    ----
    # 获取放电持续时间超过0.3秒的炮号
    shots = manager.get_shots_by_conditions("ip_feature", min_duration=0.3)
    """
    try:
        with pg_conn.cursor() as cur:
            # 1. 检查表是否存在
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table_name,))
            if not cur.fetchone()[0]:
                raise ValueError(f"表 {table_name} 不存在")

            # 2. 动态构建查询条件
            conditions = []
            params = []
            
            # 持续时间条件
            if min_duration is not None or max_duration is not None:
                duration_clause = "(end_time - start_time)"
                if min_duration is not None:
                    conditions.append(f"{duration_clause} >= %s")
                    params.append(min_duration)
                if max_duration is not None:
                    conditions.append(f"{duration_clause} <= %s")
                    params.append(max_duration)
            
            # 特征条件（仅当表有多标签组时有效）
            table_columns = get_table_columns(pg_conn, table_name)
            if "feature" in table_columns and feature_name is not None:
                operator = "=" if has_feature else "!="
                conditions.append(f"feature {operator} %s")
                params.append(feature_name)
            elif "feature" not in table_columns and feature_name is not None:
                raise ValueError("该表不支持特征筛选")
            
            # 标签条件
            label_col = label_column_name if label_column_name in table_columns else None
            if label_col and label_name is not None:
                operator = "=" if has_label else "!="
                conditions.append(f"{label_col} {operator} %s")
                params.append(label_name)
            elif label_name is not None and label_col is None:
                raise ValueError("该表不支持标签筛选")

            # 3. 组合完整查询
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            sql = f"""
                SELECT DISTINCT shot 
                FROM {table_name} 
                {where_clause}
                ORDER BY shot ASC;
            """
            cur.execute(sql, params)
            return [row[0] for row in cur.fetchall()]

    except psycopg2.Error as e:
        print(f"数据库错误: {e}")
        return []
    except ValueError as e:
        print(f"参数错误: {e}")
        return []     

def query_annotations(
    pg_conn: psycopg2.extensions.connection,
    shots: List[int],
    table_names: List[str]|str,
    columns: Optional[List[str]] = None,
    all_info: bool = False,
    schema: str = "public"
) -> List[Dict[str, Any]]:
    """
    根据给定的shots和多个标注表，从PostgreSQL中导出标注信息。
    
    参数：
    ----
    shots: List[int]
        要筛选的炮号列表。
    table_names: List[str]
        一个或多个表名（存储不同特征的标注表）。
    columns: Optional[List[str]]
        如果指定，则只返回这些列的数据；若为 None 则根据all_info决定返回全部或常用列。
    all_info: bool
        是否返回表中的所有列，默认为False。
        当columns为None且all_info=False时，默认只返回常用列；
        若columns为None且all_info=True，则返回所有列。
    schema: str
        数据库模式名，默认为“public”。
    
    返回：
    ----
    result_list: List[Dict[str, Any]]
        由字典组成的列表，每个字典代表一条标注信息，示例：
        {
        "table": "disruption_table",
        "shot": 241011005,
        "start_time": 0.55,
        "end_time": 0.60,
        "label": "major",
        "is_point": false,
        "annotator": "alice",
        ...
        }
        其中还包含其他可能的列，具体取决于 columns 和 all_info。
    
    用法示例:
    ----
    manager = AnnotationManager()
    shots = [241011005, 241012030]
    data = manager.export_annotation_data(shots, ["ip_feature", "disruption_table"],
                                        columns=["shot","start_time","end_time","label","annotation_id"],
                                        all_info=False)
    # data将包含来自ip_feature表和disruption_table表、且shot在上述列表中的全部记录
    """
    result_list = []
    if not shots or not table_names:
        return result_list
    if isinstance(table_names,str):
        table_names = [table_names]

    try:
        # 将shots列表转为set再转为list，去除重复元素，方便SQL的IN或ANY
        shot_set = list(set(shots))  
        if not shot_set:
            return result_list
        
        # 处理 columns 与 all_info 逻辑
        # 如果 columns=None 且 all_info=False，则仅返回一组“常用列”
        # 如果 columns=None 且 all_info=True，则返回表中所有列
        # 如果 columns 不为空，则只返回用户指定的列
        
        for table in table_names:
            # 1. 检查表是否存在
            with pg_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema=%s AND table_name=%s
                    );
                    """,
                    (schema, table)
                )
                table_exists = cur.fetchone()[0]
                if not table_exists:
                    print(f"警告: 表 {table} 不存在，跳过")
                    continue
            
            # 2. 如果all_info=True,则查询表头
            actual_columns = columns
            if columns is None:
                if all_info:
                    # 查询所有列
                    actual_columns = get_table_columns(pg_conn, table_name=table, schema=schema)
                else:
                    # 否则只返回常用列
                    actual_columns = []
                    possible_common_cols = [
                        "shot","label","feature","start_time","end_time",
                        "is_point","annotation_id",
                    ]
                    # 取交集
                    table_cols = get_table_columns(pg_conn, table_name=table, schema=schema)
                    for col in possible_common_cols:
                        if col in table_cols:
                            actual_columns.append(col)
                    if not actual_columns:
                        # 如果这张表连常用列都没有，跳过
                        continue
            
            # 3. 构造SQL: SELECT actual_columns FROM <table> WHERE shot in (shots) ORDER BY shot
            col_str = ", ".join(actual_columns)
            placeholders = ", ".join(["%s"] * len(shot_set))
            sql = f"""
                SELECT {col_str}
                FROM {schema}.{table}
                WHERE shot = ANY(%s)
                ORDER BY shot;
            """
            
            with pg_conn.cursor() as cur:
                cur.execute(sql, (shot_set,))
                rows = cur.fetchall()
                # 构造列名->index
                col_index_map = {col: i for i, col in enumerate(actual_columns)}
                
                for row in rows:
                    row_dict = {}
                    # 将每列的值填进字典
                    for col in actual_columns:
                        row_dict[col] = row[col_index_map[col]]
                    # 加一个字段区分所属表
                    row_dict["table"] = table
                    result_list.append(row_dict)
        
        return result_list
    except Exception as e:
        print(f'导出标注数据错误: {e}')
        return []
