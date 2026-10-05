"""Synthetic die-face generator used to produce the images in set2(jai).

Renders rounded-square dice (faces 1-6) with Pillow: random die colour (7
options), pip colour chosen for contrast, random background colour + grain,
drop shadow, random rotation (0-360 deg) and shift, optional blur, brightness
and noise jitter. Rendered at 256x256 and downsampled to 128x128.

Fixed seed (42) -> reproducible. Running it writes 3000 images
(dice_dataset/{train,val}/{face}/die{face}_{index:04d}.png); the 119 images in
set2(jai)/unprocessed/ are a sample from that output. Run from any directory:
    python3 generate_synthetic.py
Requires: pillow, numpy.
"""
import random, math, os, zipfile
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

random.seed(42); np.random.seed(42)
SIZE = 128
PIPS = {
 1: [(0,0)],
 2: [(-1,-1),(1,1)],
 3: [(-1,-1),(0,0),(1,1)],
 4: [(-1,-1),(1,-1),(-1,1),(1,1)],
 5: [(-1,-1),(1,-1),(0,0),(-1,1),(1,1)],
 6: [(-1,-1),(1,-1),(-1,0),(1,0),(-1,1),(1,1)],
}
DIE_COLORS = [(250,250,245),(220,30,30),(30,30,30),(30,90,200),(40,150,70),(240,200,40),(130,50,160)]

def contrast_pip(c):
    return (20,20,20) if sum(c)/3 > 140 else (245,245,245)

def make(face):
    S = SIZE*2  # supersample
    bg = tuple(random.randint(30,230) for _ in range(3))
    img = Image.new("RGB",(S,S),bg)
    # background texture
    arr = np.array(img).astype(np.int16)
    arr += np.random.randint(-12,12,arr.shape)
    img = Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    die = random.choice(DIE_COLORS)
    pip = contrast_pip(die)
    side = int(S*random.uniform(0.5,0.68))
    layer = Image.new("RGBA",(S,S),(0,0,0,0))
    ld = ImageDraw.Draw(layer)
    x0=(S-side)//2; y0=(S-side)//2
    r = int(side*0.16)
    # shadow
    sh = Image.new("RGBA",(S,S),(0,0,0,0))
    ImageDraw.Draw(sh).rounded_rectangle([x0+8,y0+10,x0+side+8,y0+side+10],r,fill=(0,0,0,90))
    sh = sh.filter(ImageFilter.GaussianBlur(8))
    ld.rounded_rectangle([x0,y0,x0+side,y0+side],r,fill=die+(255,),outline=tuple(max(0,c-50) for c in die)+(255,),width=3)
    pr = side*0.085
    step = side*0.27
    cx,cy = S/2,S/2
    for (px,py) in PIPS[face]:
        jx,jy = random.uniform(-1.5,1.5),random.uniform(-1.5,1.5)
        x=cx+px*step+jx; y=cy+py*step+jy
        ld.ellipse([x-pr,y-pr,x+pr,y+pr],fill=pip+(255,))
    angle = random.uniform(0,360)
    layer = layer.rotate(angle,resample=Image.BICUBIC,center=(S/2,S/2))
    sh = sh.rotate(angle,resample=Image.BICUBIC,center=(S/2,S/2))
    # random shift
    dx,dy = random.randint(-S//12,S//12),random.randint(-S//12,S//12)
    img.paste(sh,(dx,dy),sh); img.paste(layer,(dx,dy),layer)
    img = img.resize((SIZE,SIZE),Image.LANCZOS)
    if random.random()<0.4: img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3,1.2)))
    a = np.array(img).astype(np.int16)
    a = a*random.uniform(0.8,1.15) + np.random.normal(0,random.uniform(0,6),a.shape)
    return Image.fromarray(np.clip(a,0,255).astype(np.uint8))

N_TRAIN, N_VAL = 400, 100
root="dice_dataset"
for split,n in [("train",N_TRAIN),("val",N_VAL)]:
    for f in range(1,7):
        os.makedirs(f"{root}/{split}/{f}",exist_ok=True)
        for i in range(n):
            make(f).save(f"{root}/{split}/{f}/die{f}_{i:04d}.png")
# montage preview
m = Image.new("RGB",(SIZE*6,SIZE*3))
for f in range(1,7):
    for k in range(3):
        m.paste(Image.open(f"{root}/train/{f}/die{f}_{k:04d}.png"),((f-1)*SIZE,k*SIZE))
m.save("preview.png")
with zipfile.ZipFile("dice_dataset.zip","w",zipfile.ZIP_DEFLATED) as z:
    for dp,_,fs in os.walk(root):
        for fn in fs: z.write(os.path.join(dp,fn))
print("done")
