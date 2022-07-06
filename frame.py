# -*- coding: utf-8 -*-
import os,sys
import time
import numpy as np
import glob
# ffmpegのインストールは下記参照
# https://irohaplat.com/windows-10-ffmpeg-installation-procedure/
import ffmpeg

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    #[0]はpyファイル
    #print(sys.argv[0])
    log = []
    # 動画フォルダのパスを指定
    input_path = sys.argv[1]
    os.chdir(os.path.abspath(input_path))
    print(os.path.abspath(input_path))
    # folders = sorted(glob.glob("./*"))
    # #フォルダ名（ファイル名）を拡張子を切って取得
    # folder_names = [os.path.splitext(os.path.basename(folder))[0] for folder in folders ]
    # for i in range(len(folder_names)):
    #     files = sorted(glob.glob("./"+folder_names[i]+"/*.mp4" ))   

    files = sorted(glob.glob("./*.MP4"))
    if len(files)==0:
        print("なし")
    for ii in range(len(files)):
        file_name = os.path.splitext(os.path.basename(files[ii]))[0]
        save_path = "../frame/"+file_name
        os.makedirs(save_path, exist_ok=True)
        #動画の時間（duration,秒単位)を取得する
        print(files[ii])
        video_info = ffmpeg.probe(files[ii])
        duration = float(video_info["streams"][0]["duration"])
        if duration > 3610.0:
            duration = 3610.0
        for iii in range(1, int(duration)+1):
            #既に作成済みの場合スルー
            if os.path.exists(save_path+"/"+file_name+"_"+str(iii-1).zfill(5)+".png") ==True:
                log.append(file_name+"_"+str(iii-1).zfill(5)+".png")
                continue
            #動画からフレーム切り出しの開始時間ss（秒単位）にループの変数を与える（1~3610秒間）
            stream = ffmpeg.input(files[ii], ss =iii)
            stream = ffmpeg.output(stream, save_path+"/"+file_name+"_"+str(iii-1).zfill(5)+".png", c = "copy", vcodec = "png", vframes = 1)
            try:
                #上記のstreamを実行
                ffmpeg.run(stream)
                log.append(file_name+"_"+str(iii-1).zfill(5)+".png")
            except:
                log.append(file_name+"_"+str(iii-1).zfill(5)+".png error")

    os.makedirs("../log", exist_ok=True)
    now = time.localtime()
    date = time.strftime('%Y%m%d%H',now)  
    with open("../log/"+date+"_frame_log.txt", mode="w", encoding="utf-8") as f:
        f.writelines("\n".join(log))
        f.close()

if __name__ == '__main__':
    main()


