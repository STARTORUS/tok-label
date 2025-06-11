
# 测试文件服务器
# 1. 列出文件
# 2. 从pgsql导出数据
# %%
import requests
import json
# 服务器地址
server_url = "http://dap0.lan:30422"
#server_url = "http://127.0.0.1:8000"
# 项目名称
params = {
    "dir": "test"
}
# 列出文件
response = requests.get(server_url + "/list", params=params, verify=False)
# 解析返回的文件列表    
data = json.loads(response.text)
print(data)


# %%

# 从pgsql导出数据
url = server_url + "/export"

# 定义需要导出的表和列
name_table_columns_demo = {
        "ammeter": ("ammeter", ["CS1", "CS2", "PFP1"]),
        "flux_loop": ("flux_loop", [1, 2, 3, 8, 10])}
# 定义炮号
shots=[240830026, 240830027, 240830028, 240830029]
# 定义时间范围
t_min = 0
t_max = 1
# 定义降采样分辨率
resolution = 1e-3
# 定义请求参数
params = {
    "project_name": "test",
    "shots": shots,
    "name_table_columns": name_table_columns_demo,
    "t_min": t_min,
    "t_max": t_max,
    "resolution": resolution
}
response = requests.post(url, json=params, verify=False)
print(response.text)


# %%

