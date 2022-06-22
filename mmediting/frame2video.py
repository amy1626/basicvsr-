#coding=utf-8
import glob

import cv2
import numpy as np

#把 帧 合并成 视频
def frame2video(frame_dir,video_dir,fps,w,h):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowrite=cv2.VideoWriter(video_dir,fourcc,fps,(w,h))
    path_frame=sorted(glob.glob(f'{frame_dir}/*'))
    for i,name in enumerate(path_frame):
        img=cv2.imread(name)#读取图片
        videowrite.write(img)
    cv2.destroyAllWindows()
    videowrite.release()

if __name__=='__main__':
    
    frame_dir='data/reds/traffic/test/gt/000'
    video_dir='work_dirs/generate_video/000gt.mp4'
    w=1920
    h=1080
    fps=25

    frame2video(frame_dir,video_dir,fps,w,h)
    print('-----------end!!--------------')