import cv2

from data_loader import BSDDB


if __name__ == "__main__":
    db = BSDDB(folder_dir="/Users/starry/ualberta/TemplateMatching/PACKAGEJIAXIN", ext=".jpg", patch_size=80, fake_legth=1000)
    # db = BSDDB(folder_dir="/home/yusun/mmlab/datasets/BSR/BSDS500/data/images/train",
    #            ext=".jpg",
    #            patch_size=80,
    #            fake_legth=1000)
    print(db[0].size)
    cv2.imshow("img_patch[0]", db[0])
    cv2.waitKey(0)


