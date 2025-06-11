# %%
# test functions in toklabel.utils.py

from toklabel.utils import export_data, export_urls_to_redis

# %%
project_name = "test"
shots = [240830026, 240830027, 240830028, 240830029]
name_table_columns = {
        "ammeter": ("ammeter", ["CS1", "CS2", "PFP1"]),
        "flux_loop": ("flux_loop", [1, 2, 3, 8, 10])
        }

# 导出数据
result = export_data(
    project_name=project_name,
    shots=shots,
    name_table_columns=name_table_columns,
)
urls = result["urls"]
print(result)
# %%
# 导出urls到redis
result = export_urls_to_redis(
    project_name=project_name,
    urls=urls,
    key="csv"
)
print(result)
# %%
from toklabel.utils import connect_redis
# list redis keys
redis_conn = connect_redis()
keys = redis_conn.keys()
print(keys)

# %%
