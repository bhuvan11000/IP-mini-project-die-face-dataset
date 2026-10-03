#!/bin/bash
DEST=/home/bhu1/sem5/ip/IP-mini-project-die-face-dataset/set1
SRC=/sdcard/DCIM/Camera
mkdir -p "$DEST"
prev=""

for g in {1..6}; do
  for i in {0..9}; do
    name="${g}_${i}.jpg"
    read -p "Ready for $name? Press Enter to shoot (Ctrl+C to quit) "

    while true; do
      adb shell input keyevent KEYCODE_CAMERA
      sleep 2
      latest=$(adb shell "ls -t $SRC/*.jpg | head -n1" | tr -d '\r')
      if [ -n "$latest" ] && [ "$latest" != "$prev" ]; then
        adb pull "$latest" "$DEST/$name" && prev="$latest"
        break
      fi
      echo "No new photo detected, retrying..."
    done
  done
done
