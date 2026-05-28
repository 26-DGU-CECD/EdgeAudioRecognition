jetson 실행 명령어
```
cd ~

git clone -b playground https://github.com/26-DGU-CECD/EdgeAudioRecognition.git

rm -rf ~/efficientat_ws
cp -r ~/EdgeAudioRecognition/playground/yunyeong/efficientat_ws ~/efficientat_ws

cd ~/efficientat_ws
chmod +x run_wifi_bridge_finetuned_from_host.sh

./run_wifi_bridge_finetuned_from_host.sh plughw:2,0 8765
```

스크립트는 기본적으로 ~/efficientat_ws가 존재한다고 가정하기 때문에 반드시 host에 efficient_ws 파일 복사해야 함 
