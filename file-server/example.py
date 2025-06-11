import requests
import json
from typing import Dict, List, Union

def list_files(server_url: str, directory: str = None) -> Dict:
    """
    列出服务器上的文件
    
    参数:
    server_url (str): 服务器基础地址 (例如 "http://localhost:8000")
    directory (str, optional): 要列出的目录路径

    返回:
    Dict: 包含文件和URL列表的字典
    """
    try:
        params = {"dir": directory} if directory else None
        response = requests.get(f"{server_url}/list/", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def export_data(
    server_url: str,
    project_name: str,
    shots: Union[int, List[int]],
    name_table_columns: Dict,
    t_min: float = None,
    t_max: float = None,
    resolution: float = None
) -> Dict:
    """
    导出PostgreSQL数据到CSV文件
    
    参数:
    server_url (str): 服务器基础地址
    project_name (str): 项目名称（用于文件存储）
    shots (Union[int, List[int]]): 单个或多个炮号
    name_table_columns (Dict): 要导出的表和列定义
    t_min (float, optional): 起始时间
    t_max (float, optional): 结束时间 
    resolution (float, optional): 时间分辨率

    返回:
    Dict: 包含操作结果和文件URL的字典
    """
    try:
        payload = {
            "project_name": project_name,
            "shots": shots,
            "name_table_columns": name_table_columns,
            "t_min": t_min,
            "t_max": t_max,
            "resolution": resolution
        }
        
        response = requests.post(
            f"{server_url}/export/",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def delete_file(server_url: str, directory: str, file: str = None) -> Dict:
    """
    删除文件
    
    参数:
    server_url (str): 服务器基础地址
    directory (str): 目录路径
    file (str, optional): 文件名

    返回:
    Dict: 包含操作结果的字典
    """
    try:
        params = {"dir": directory, "file": file} if file else {"dir": directory}
        response = requests.delete(f"{server_url}/delete/", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # 示例用法
    SERVER_URL = "http://dap0.lan:30422"
    # SERVER_URL = "http://localhost:8000"
    
    # 示例1：列出文件
    print("列出文件示例:")
    file_list = list_files(SERVER_URL, "demo_project")
    print(json.dumps(file_list, indent=2, ensure_ascii=False))
    
    # 示例2：导出数据
    print("\n导出数据示例:")
    export_result = export_data(
        server_url=SERVER_URL,
        project_name="demo_project",
        shots=[240830026, 240830027],
        name_table_columns={
            "ammeter": ["ammeter", ["CS1", "PFP1"]],
            "flux_loop": ["flux_loop", [1, 2, 8]]
        },
        t_min=0.0,
        t_max=1.0,
        resolution=0.001
    )
    print(json.dumps(export_result, indent=2, ensure_ascii=False))

    # 示例3：删除文件
    print("\n删除文件示例:")
    delete_result = delete_file(SERVER_URL, "demo_project")
    print(json.dumps(delete_result, indent=2, ensure_ascii=False))
