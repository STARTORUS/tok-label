import psycopg2
import psycopg2.sql as sql
from psycopg2.extras import execute_values
from ..config import POSTGRE_HOST, POSTGRE_PORT, POSTGRE_USER, POSTGRE_PASSWORD, POSTGRE_DATABASE
from sunist2.script.postgres import execute_query, query_data
from typing import Optional, Dict, Any, List, Sequence, Tuple
from datetime import datetime
import pytz

def connect_annotation_database():
    conn = psycopg2.connect(
        host=POSTGRE_HOST,
        port=POSTGRE_PORT,
        database=POSTGRE_DATABASE,
        user=POSTGRE_USER,
        password=POSTGRE_PASSWORD
    )
    return conn

def list_existing_tables(pg_conn: psycopg2.extensions.connection, schema:int|str = 'public') -> List[str]:
    """
    列出当前数据库中的所有表名，供用户查看是否已存在某特征表

    参数：
    pg_conn：Postgresql客户端
    """
    sql_command = sql.SQL(("""SELECT tablename 
                FROM pg_catalog.pg_tables 
                WHERE schemaname = {};""").format(sql.Literal(str(schema))))
    with pg_conn.cursor() as cur:
        cur.execute(sql_command)
        res = cur.fetchall()
    return [r[0] for r in res]
    
def get_table_columns(
    pg_conn,
    table_name: str,
    schema: str | int = "public",
    name_only:bool =True,
    include_type: Optional[Sequence[str]] = None,   # 仅这些类型
    exclude_columns: Optional[Sequence[str]] = None    # 排除这些列名
):
    """
    返回表的列及其 PostgreSQL 数据类型。
    include:   仅保留 data_type 在该列表内的列 (如 ['double precision','numeric'])
    exclude:   列名排除表
    """
    q = sql.SQL("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = {schema}
          AND table_name   = {table}
        ORDER BY ordinal_position;
    """).format(
        schema = sql.Literal(str(schema)),
        table  = sql.Literal(table_name)
    )

    with pg_conn.cursor() as cur:
        cur.execute(q)
        cols = cur.fetchall()          # [(name, type), ...]

    if include_type:
        include_type = set(t.lower() for t in include_type)
        cols = [(c, t) for c, t in cols if t.lower() in include_type]

    if exclude_columns:
        exclude_columns = set(exclude_columns)
        cols = [(c, t) for c, t in cols if c not in exclude_columns]

    return [c for c, _ in cols] if name_only else cols

def delete_table(
        pg_conn: psycopg2.extensions.connection, 
        table_name: str, 
        schema:str|int ='public'
):
    """
    删除数据库中的表

    参数:
    ----
    table_name :str 表名
    schema :str 默认为"public"
    """
    schema = f'"{schema}"'
    table = f'"{table_name}"'
    cur = pg_conn.cursor()
    # 如果不存在schema，则创建schema
    cur.execute(f"""
    CREATE SCHEMA IF NOT EXISTS {schema};
    """)
    # 删除表
    cur.execute(f"""
    DROP TABLE IF EXISTS {schema}.{table};
    """)
    pg_conn.commit()
    cur.close()        

def string_to_timestampz(
            date_time_str:str, 
            timezone="Asia/Shanghai"
        ):
        """
        将输入的时间字符串转换为timestampz格式
        """
        if date_time_str is None:
            return None
        dt_utc = datetime.strptime(date_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        dt_utc = dt_utc.replace(tzinfo=pytz.utc)
        # 转为上海时区（Asia/Shanghai）
        shanghai_tz = pytz.timezone(timezone)
        dt_shanghai = dt_utc.astimezone(shanghai_tz)
        return dt_shanghai

def delete_shot_schema(
        pg_conn: psycopg2.extensions.connection, 
        shot: int):
    '''
    删除某个炮号对应的schema的全部数据，谨慎使用
    '''
    schema_idt = sql.Identifier(str(shot))
    
    with pg_conn.cursor() as cur:
        cur.execute(sql.SQL
                    ("DROP SCHEMA IF EXISTS {schema} CASCADE")
                    .format(schema=schema_idt))