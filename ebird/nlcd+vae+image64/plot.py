from PIL import Image
import numpy as np

data = np.load('tsne_data.npy')

base_img = Image.new("RGB",(700,700),(255,255,255))

forest_order = [0,1,2,3,12,19,38,75,79,147]

human_order = [238,257,726,886,888,1397,1730,1910,26834,27110]

ocean_order = [45,178,266,876,1516,8112,8201,8365,8495,9318,26471]

orders = forest_order+human_order+ocean_order
print len(orders)

for i in range(len(orders)):

    x = int((150+data[i][0])*2)
    y = int((150+data[i][1])*2)

    box = (x, y, x+60, y+60)

    color = []
    if i <10:
        color = (0,255,0)
    if i >=10 and i<20:
        color = (190,190,190)
    if i >=20:
        color = (0,0,255)
    tang_base = Image.new('RGB',(60,60),color)
    tmp_img = Image.open('../new_crop_pic/'+str(orders[i])+'.jpg')

    region = tmp_img


    region = region.resize((50,50))
    tang_base.paste(region,(5,5,55,55))
    base_img.paste(tang_base, box)

base_img.save('./out.png')