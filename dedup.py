"""
图片去重
"""
import os
from imagededup.methods import PHash

def process_file(img_path):
    """
    处理图片去重
    :return:
    """
    try:
        phasher = PHash()
        # 生成图像目录中所有图像的二值hash编码
        encodings = phasher.encode_images(image_dir=img_path)
        # print(encodings)
        # 对已编码图像寻找重复图像
        duplicates = phasher.find_duplicates(encoding_map=encodings)
        # print(duplicates)
        only_img = []  # 唯一图片
        like_img = []  # 相似图片

        for img, img_list in duplicates.items():
            if ".png" in img:
                continue
            if img not in only_img and img not in like_img:
                only_img.append(img)
                like_img.extend(img_list)

        # 删除文件
        for like in like_img:
            like_src = os.path.join(img_path, like)
            png_src = like_src[:-4] + ".png"
            if os.path.exists(like_src):
                os.remove(like_src)
            if os.path.exists(png_src):
                os.remove(png_src)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    img_path = "D:/EmojiDataset"

    num = 0
    for root, dirs, files in os.walk(img_path):
        for dir in dirs:
            file_dir_path = os.path.join(root, dir)
            process_file(file_dir_path)
            num += 1
            print("处理文件夹个数:{}".format(num))
