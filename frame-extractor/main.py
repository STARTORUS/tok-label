from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
from typing import Optional
from pydantic import BaseModel, Field
from sunist2.script.camera import get_video
from typing import Dict, List
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("/data")
URL_PREFIX = "/files"

class ExtractRequest(BaseModel):
    project_name: str
    shots: int|list[int]
    t_min: float|List[float] = -np.inf
    t_max: float|List[float] = np.inf
    resolution: float = Field(1e-3, gt=1e-5, description = "time resolution in s")

SHOTS_URL :Dict[int, List[str]] = {}
SHOTS_FRAMETIME :Dict[int,List[float]] = {}

def save_frames(project_name :str, shot: int, t_min: float, t_max: float, resolution: float=1e-4):
    """Background task: extract frames for one shot and save as JPGs."""
    url_list = []
    try:
        t, frames = get_video(shot, t_min, t_max, resolution=resolution)
        if t is None:
            raise RuntimeError("get_video returned None")
        outdir = DATA_DIR/ f"{project_name}/{shot}"
        outdir.mkdir(parents=True, exist_ok=True)
        t0 = t[0]
        for tt, frame in zip(t, frames):
            t_stamp = int(round(tt * 1e5))
            fname = f"{t_stamp:06d}.jpg"
            cv2.imwrite(str(outdir/ fname), frame)
            url_list.append(f"{URL_PREFIX}/{project_name}/{shot}/{fname}")
    finally:
            SHOTS_URL[shot] = url_list
            SHOTS_FRAMETIME[shot] = t.tolist()


@app.post("/extract/frames")
async def extract_frames(
    request: ExtractRequest,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    异步提帧
    """
    if isinstance(request.shots, int):
        request.shots = [request.shots]
    if not isinstance(request.t_min, list):
        request.t_min = [request.t_min]
    if not isinstance(request.t_max, list):
        request.t_max = [request.t_max]    
    if len(request.t_min) != len(request.shots) or len(request.t_max) != len(request.shots):
        raise ValueError('lengths of t_min, t_max and shots are not equal')   
    #for shot in request.shots:
    #    background_tasks.add_task(save_frames, 
    #                              request.project_name, 
    #                              shot, 
    #                              request.t_min, request.t_max, 
    #                              request.resolution)
    for shot, t_min, t_max in zip(request.shots,request.t_min, request.t_max):
        save_frames(request.project_name, shot, t_min, t_max, request.resolution)    
    return {"message": "images extracted successfully", 
            "urls": SHOTS_URL,
            "frame_time": SHOTS_FRAMETIME}
