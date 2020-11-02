import cv2
import os

def bgr2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def histEqualize(img,mode='clahe',space='gray',clipLimit=20.0):
    """
    equalize histogram of a image.
    :param mode: if norm,perform normal hist, if clahe,perform adaptive histogram equalization.
    :return:
    """
    if space == 'gray':
        if mode =='norm':
            return cv2.equalizeHist(img)
        elif mode =='clahe':
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
            return clahe.apply(img)
    elif space == 'rgb':
        if mode == 'norm':
            raise  NotImplementedError
        elif mode =='clahe':
            img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            clahe=cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=(8,8))
            value=img[:,:,2]
            value=clahe.apply(value)
            img[:,:,2]=value
            img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
            return img

def img2jpg(src_path,dst_path):
    for i in range(1,8):
        src_dir=os.path.join(src_path,'Camera{}'.format(i))
        dst_dir =os.path.join(dst_path,'Camera{}'.format(i))
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        for file in os.listdir(src_dir):
            img_path=os.path.join(src_dir,file)
            out_path=os.path.join(dst_dir,file[:-3]+'jpg')
            img=cv2.imread(img_path)
            cv2.imwrite(out_path,img)

def read_txt_mklist(txt_path):
    bboxes_list=[]
    files=os.listdir(txt_path)
    files.sort()
    for txt in files:
        file=os.path.join(txt_path,txt)
        with open(file,'r')as f:
            bboxes_Camera=[]
            bboxes=f.readlines()
            for i,box in enumerate(bboxes):
                box=box.rstrip().split(' ')
                if len(box)==1 and '' in box:
                    box=[]
                else:
                    box=[int(i) for i in box]
                bboxes[i]=box
        bboxes_list.append(bboxes)

    batches=[]
    for i in range(len(bboxes_list[0])):
        batch = []
        for j in range(len(bboxes_list)):
            box=bboxes_list[j][i]
            bboxes_tmp=[]
            for k in range(len(box) // 4):
                bboxes_tmp += [box[k * 4:k * 4 + 4]]
            batch.append(bboxes_tmp)
        batches.append(batch)

    return batches

def shift_bboxes_to_stitch(bboxes,offset_w):
    """
    add offset w to bboxes to get stitched bboxes.
    :param bboxes: (list(list)), lots of bboxes.
    :param offset_w: (int) offset w.
    :return:
    """
    if len(bboxes) ==0:
        return bboxes
    for i in range(len(bboxes)):
        xmin,ymin,xmax,ymax,score=bboxes[i]
        xmin_new=xmin+offset_w
        xmax_new=xmax+offset_w
        bboxes[i]=[xmin_new,ymin,xmax_new,ymax,score]
    return bboxes

def draw_bboxes(img,bboxes):
    for box in bboxes:
        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color=(0,0,255),thickness=2)
    return img

def reshape_bboxes(bboxes):
    box=[]
    for i in bboxes:
        for j in i:
          box+=[j]
    return box

# path = "/data2/yeliang/dataset/stitch_test/bbox"
# array=read_txt_mklist(path)
# # array=np.array(array)
# # print(array.shape)
# box=reshape_bboxes(array)

