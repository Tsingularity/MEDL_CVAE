from PIL import Image
import numpy as np

data = np.load('tsne_data.npy')

base_img = Image.new("RGB",(700,700),(255,255,255))

forest_order = [0,1,2,3,12,19,38,75,79,147,13266,12226,12316,12341,12514,14090,14649,19237,19390,20048]

human_forest_order = [238,257,726,886,888,1397,1730,26834,27110,13219,13147,16077,18200,18633,18849,19534,19941,19925,43478]

human_ocean_order = [1910,16365,16366,17355,18085,19758,44284,44691,44842,45,266,876,14883,15116,50694]

ocean_order = [178,1516,8112,8201,8365,8495,9318,26471,14503,14550,14567,15025,17391,18230,20560,20534,42954]

orders = forest_order+human_forest_order+human_ocean_order+ocean_order
print len(orders)

for i in range(len(orders)):

    x = int((150+data[i][0])*2)
    y = int((150+data[i][1])*2)

    box = (x, y, x+60, y+60)

    color = []
    if i <len(forest_order):
        color = (0,255,0)
    if i >=len(forest_order) and i< (len(forest_order)+len(human_forest_order)+len(human_ocean_order)):
        color = (190,190,190)
    if i >=(len(forest_order)+len(human_forest_order)+len(human_ocean_order)):
        color = (0,0,255)
    tang_base = Image.new('RGB',(60,60),color)
    tmp_img = Image.open('../new_crop_pic/'+str(orders[i])+'.jpg')

    region = tmp_img


    region = region.resize((50,50))
    tang_base.paste(region,(5,5,55,55))
    base_img.paste(tang_base, box)

base_img.save('./out.png')