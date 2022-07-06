# -*- coding: utf-8 -*-
import os,sys
import time
import cv2
import numpy as np
import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    #[0]はpyファイル
    #print(sys.argv[0])
    log = []
    #frameフォルダを指定する
    input_path = sys.argv[1]
    os.chdir(os.path.abspath(input_path))
    
    folders = sorted(glob.glob("./*"))
    if len(folders)==0:
        print("なし")
    #フォルダ名（ファイル名）を拡張子を切って取得
    folder_names = [os.path.splitext(os.path.basename(folder))[0] for folder in folders ]
    for i in range(len(folder_names)):
        files = sorted(glob.glob("./"+folder_names[i]+"/*.png" ))        
        os.makedirs("../average/"+folder_names[i], exist_ok=True)
        #フレーム画像の作成は、1秒ごとの3610枚（1時間分ちょっと）で止めているため、50分～の10分間平均化画像があるかチェックし、あればスルー
        if os.path.exists("../average/"+folder_names[i]+"/"+folder_names[i]+"_"+str(3000).zfill(5)+"_"+str(3599).zfill(5)+ ".png")==True:
            print("平均化画像作成済")
            continue
        else:
            for ii in range(0, len(files), 600):
                first = os.path.splitext(os.path.basename(files[ii:ii+600][0]))[0].split("_")[-1]
                last = os.path.splitext(os.path.basename(files[ii:ii+600][-1]))[0].split("_")[-1]
                if os.path.exists("../average/"+folder_names[i]+"/"+folder_names[i]+"_"+str(first).zfill(5)+"_"+str(last).zfill(5)+".png") == True:
                    log.append(folder_names[i]+"_"+str(first).zfill(5)+"_"+str(last).zfill(5)+".png")
                    continue
                try:
                    average = MultiImageMeanBGR(files[ii:ii+600])
                except:
                    log.append(folder_names[i]+"_"+str(first).zfill(5)+"_"+str(last).zfill(5)+".png error")
                    continue
                cv2.imwrite("../average/"+folder_names[i]+"/"+folder_names[i]+"_"+str(first).zfill(5)+"_"+str(last).zfill(5)+".png", average)
                log.append(folder_names[i]+"_"+str(first).zfill(5)+"_"+str(last).zfill(5)+".png")
    
    os.makedirs("../log", exist_ok=True)
    now = time.localtime()
    date = time.strftime('%Y%m%d%H',now)  
    with open("../log/"+date+"_average_log.txt", mode="w", encoding="utf-8") as f:
        f.writelines("\n".join(log))
        f.close()

#入力：画像パスを格納したリスト(list)、出力：入力リストの画像をbgr毎に単純平均化した画像（array）
def MultiImageMeanBGR(image_path_list):
    #リスト内の画像のarrayを取得（出力用）
    output = cv2.imread(image_path_list[0]).astype(np.float32)
    #入力のリスト内のパスの画像をbgr毎にリストに格納
    #cv2.imreadをbgrそれぞれ行っており処理が重いので改善の余地有
    image_list_b = [cv2.imread(image_path)[:,:,0].astype(np.float32) for image_path in image_path_list]
    image_list_g = [cv2.imread(image_path)[:,:,1].astype(np.float32) for image_path in image_path_list]
    image_list_r = [cv2.imread(image_path)[:,:,2].astype(np.float32) for image_path in image_path_list]
    #画素毎にチャネル方向に足し合わせ（sum）、画像枚数で割る
    resb = sum(image_list_b)/ len(image_list_b)
    resg = sum(image_list_g)/ len(image_list_g)
    resr = sum(image_list_r)/ len(image_list_r)
    #最初に作成したarrayを置き換える
    output[:,:,0] = resb
    output[:,:,1] = resg
    output[:,:,2] = resr
    return output

if __name__ == '__main__':
    main()


