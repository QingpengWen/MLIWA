import os
import datetime
import requests
from bs4 import BeautifulSoup as be
from PIL import Image, ImageDraw, ImageFont, ImageFilter

md_url = "https://ys.mihoyo.com/content/ysCn/getContentList?pageSize=20&pageNum=1&order=asc&channelId=150"
ly_url = "https://ys.mihoyo.com/content/ysCn/getContentList?pageSize=20&pageNum=1&order=asc&channelId=151"
dq_url = "https://ys.mihoyo.com/content/ysCn/getContentList?pageSize=20&pageNum=1&order=asc&channelId=324"
xm_url = "https://ys.mihoyo.com/content/ysCn/getContentList?pageSize=20&pageNum=1&order=asc&channelId=350"
dir_path = './data2'

def get_json(_url_):
    req = requests.get(url=_url_)
    if req.status_code == 200:
        return req.json()['data']
    else:
        return None

def clean_data(_data_):
    returns = []
    for key in _data_['list']:
        ext = key["ext"]
        data = {key['title']: {
            "角色ICON": ext[0]["value"][0]["url"],
            "电脑端立绘": ext[1]["value"][0]["url"],
            "手机端立绘": ext[15]["value"][0]["url"],
            "角色名字": key['title'],
            "简介": be(ext[7]["value"], "lxml").p.text.strip(),
            "台词": ext[8]["value"][0]["url"],
        }
        }
        returns.append(data[key['title']])
    return returns

def data():
    _json_ = {}
    for url in [md_url, ly_url, dq_url, xm_url]:
        jsonlist = clean_data(get_json(url))
        for json in jsonlist:
            _json_[json['角色名字']] = json
    return _json_

# TODO:所有数据的集合
json_all = data()
with open("./data2/al.txt", "a") as f:
    print(json_all, file=f)
def lookup(name):
    json = json_all[name]
    # json = json_all
    print("查找角色：", name)
    list_path = os.path.join(dir_path, name + ".txt")
    with open(list_path, "a") as f:
        for key, value in json.items():
            if key == "音频":
                for keys, values in json[key].items():
                    print(f"{keys}：{values}", file=f)
                    print(f"{keys}：{values}")
            else:
                print(f"{key}：{value}", file=f)
                print(f"{key}：{value}")

list(1) = ['琴', '安柏', '丽莎', '凯亚', '芭芭拉', '迪卢克', '雷泽', '温迪', '可莉', '班尼特', '诺艾尔', '菲谢尔', '砂糖', '莫娜', '迪奥娜', '阿贝多', '罗莎莉亚',
         '优菈', '埃洛伊', '米卡', '魈', '北斗', '凝光', '香菱', '行秋', '重云', '刻晴', '七七', '达达利亚', '钟离', '辛焱', '甘雨', '胡桃', '烟绯', '申鹤'
         ]
list(2) = ['云堇', '夜兰', '瑶瑶', '白术', '神里绫华', '枫原万叶', '宵宫', '早柚', '雷电将军', '九条裟罗', '珊瑚宫心海', '托马', '荒泷一斗', '八重神子',
         '神里绫人', '久岐忍', '鹿野院平藏', '提纳里', '柯莱', '多莉', '赛诺', '妮露', '纳西妲', '莱依拉', '流浪者', '珐露珊', '艾尔海森', '迪希雅', '卡维'
         ]  # '五郎',
list(3) = ['可莉', '神里绫华', '纳西妲']

for name in list():
    lookup(name)
print('Finished list()')

def generate_img(tmp_file):
    img = Image.open("data1/纳西妲0.png")
    draw = ImageDraw.Draw(img) #生成绘制对象draw
    typeface = ImageFont.truetype('simkai.ttf', 18)

    with open(tmp_file) as f:
        txt = f.read()

    draw.text((30, 245), txt, fill=(130, 0, 60),
    font=typeface)
    img.show()

    return img

timestamp_str_format = '%Y%m%d_%H%M%S'
file_prefix = '纳西妲'
def generate_img_filename():
    ts = datetime.datetime.now().strftime(timestamp_str_format)
    filename = file_prefix
    fullpath = os.path.join('.', 'data1', filename + '.png')
    return ts, fullpath

def draw_image():
    ts, filepath = generate_img_filename()
    tmp_file = 'data1/' + file_prefix + '.text'
    img = generate_img(tmp_file)
    img.save(filepath)

def image_processing():
    image_path = 'data1/纳西妲.png'
    im_list = []
    #1. read the image
    im = Image.open(image_path)
    # print (im)
    #2. generate thumb file
    im_size = (512, 512)
    im.thumbnail(im_size)
    # im_list.append(im_thumb)
    im_RGB = im.convert('RGB')
    #3. 描边
    im_list.append(im_RGB.filter(ImageFilter.CONTOUR))
    #4. 边缘强化
    im_list.append(im_RGB.filter(ImageFilter.EDGE_ENHANCE))
    #5. 浮雕
    im_list.append(im_RGB.filter(ImageFilter.EMBOSS))
    #6. 平滑
    im_list.append(im_RGB.filter(ImageFilter.SMOOTH))
    #7. 锐化
    im_list.append(im_RGB.filter(ImageFilter.SHARPEN))
    #8. 锐化遮罩
    im_list.append(im_RGB.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)))
    #9. 高斯模糊
    im_list.append(im_RGB.filter(ImageFilter.GaussianBlur(radius=2)))
    im.save('data1/纳西妲1.png')

    for im_id, im in enumerate(im_list):
        im.save('data1/纳西妲2'+ str(im_id) + '.png')
    im.save('data1/纳西妲1.gif', save_all=True, append_images=im_list)

def test():
    string_s = '123123123djaskdjkasdkl'
    bytes_s = string_s.encode(encoding='utf-8')
    print(bytes_s)

draw_image()
# image_processing()
# test()
print("pass")