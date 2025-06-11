import sunist2.script.postgres as pg
import pandas as pd
from typing import List
from toklabel.utils import upload_dataframe, export_urls_to_redis
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)    

class Dataset:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.raw_data_config = {
            "name_table_columns": 
                {
                    "ip": ("rogowski", ["of1"]),
                    "ip_eddy": ("rogowski_eddy", ["of1"])
                },
            "t_min": 0.5,
            "t_max": 0.8,
            "resolution": 1e-4
        }
        
    def load_data(self, shot: int):
        # get raw data
        conn = pg.connect_to_database()
        t, data = pg.load_data(
            conn=conn, shot=shot, 
            name_table_columns=self.raw_data_config["name_table_columns"], 
            t_min=self.raw_data_config["t_min"], 
            t_max=self.raw_data_config["t_max"], 
            resolution=self.raw_data_config["resolution"]
        )
        pg.close_connection(conn)
        # get ip data

        return {
            "time": t,
            "ip": (data["ip"] - data["ip_eddy"]).squeeze(),
        }
    
    def filter_data(self, data: dict):
        return data["ip"].max() < 100e3
    
    def process_data(self, shot: int):
        try:
            data = self.load_data(shot)
            if self.filter_data(data):
                return None
        except Exception as e:
            logger.error(f"Error loading data for shot {shot}: {e}")
            return None
        # convert the dict to pd dataframe
        df = pd.DataFrame.from_dict(data)
        # upload to file server
        result = upload_dataframe(dir=self.project_name, file=f"{shot}.csv", dataframe=df)
        if result.get("error", None) is not None:
            logger.error(f"Error uploading dataframe for shot {shot}: {result['error']}")
            return None
        url = result["url"]
        return url
    
    def create_dataset(self, shots: List[int]):
        urls = {}
        # process data in parallel
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(self.process_data, shot) for shot in shots]
            for shot, future in zip(shots, futures):
                url = future.result()
                if url is not None:
                    urls[shot] = url
        # save to redis
        result = export_urls_to_redis(project_name=self.project_name, urls=urls)
        return result
    

if __name__ == "__main__":
    conn = pg.connect_to_database()
    shots = pg.shot_range(conn=conn, shot_min=240829001, shot_max=250220001)
    pg.close_connection(conn)
    # print(shots)
    ip_features = Dataset(project_name="ip_features")
    result = ip_features.create_dataset(shots)
    print(result)
           
    
    
