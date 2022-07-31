from util.plot_utils import plot_logs
from pathlib import Path
import torch
from matplotlib import pyplot as plt
from PIL import Image
import torchvision.transforms as T
# from models import build_model
# from train import args

# 读取分类标签
with open("mask_outdoor_classes.txt", "r")as f:
    finetuned_classes = f.read().split("\n")
# 类别数
num_classes = len(finetuned_classes)

# 对输入的图片进行resize和标准化
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 定义框的颜色
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# 将模型输出的box的表示由xmid,ymid,width,height 转化为 xmin,ymin,xmax,ymax方便画图
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# 根据图片尺寸对框进行等比例变化以适应
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# 根据阈值对输出的框进行过滤
def filter_bboxes_from_outputs(outputs,
                               threshold, im):
    # keep only predictions with confidence above threshold
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    probas_to_keep = probas[keep]

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    return probas_to_keep, bboxes_scaled

# 将框，分类，score等信息绘制到图片上并显示和保存
def plot_finetuned_results(pil_img, prob=None, boxes=None):
    global img_name
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
    plt.savefig(img_name.split("/")[-1].replace(".jpg","")+"_result.jpg")

# 用模型对图片进行预测，并绘制框，类别等信息
def run_worflow(my_image, my_model):
    # 对图片进行处理以便与输入到模型
    img = transform(my_image).unsqueeze(0)
    start_time = time.time()
    # 使用模型进行预测
    outputs = my_model(img)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(total_time_str)
    # 设定阈值对框进行过滤
    for threshold in [0.6]:
        probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs,
                                                                   threshold=threshold,im=my_image)
        # 将符合条件的框绘制到图片上
        plot_finetuned_results(my_image,probas_to_keep,bboxes_scaled)

def get_log_figure():
    # 将训练过程的loss，map变化曲线进行绘制
    log_directory = [Path('outputs/')]

    fields_of_interest = (
        'loss',
        'mAP',
    )
    plot_logs(log_directory,
              fields_of_interest)

    # 将ce，bbox，giou等loss变化曲线进行绘制
    fields_of_interest = (
        'loss_ce',
        'loss_bbox',
        'loss_giou',
    )
    plot_logs(log_directory,
              fields_of_interest)

    # 对训练过程中的分类错误率变化进行绘制
    fields_of_interest = (
        'class_error',
        'cardinality_error_unscaled',
    )

    plot_logs(log_directory,
              fields_of_interest)


def video_predict(video_pth=None):
    import cv2
    from time import sleep
    model = torch.hub.load('facebookresearch/detr',
                           'detr_resnet50',
                           pretrained=False,
                           num_classes=5)

    # 读取自己训练好的模型权重
    checkpoint = torch.load('outputs/outdoor_0623.pth',
                            map_location='cpu')

    model.load_state_dict(checkpoint['model'],
                          strict=False)

    model.eval()

    cap = cv2.VideoCapture(video_pth)
    if cap.isOpened():
        ret, frame = cap.read()
        while ret:
            # cv2.imshow("frame", frame)
            cv2.waitKey(1)
            ori_img = Image.fromarray(frame)
            # 对图片进行处理以便与输入到模型
            img = transform(ori_img).unsqueeze(0)
            # 使用模型进行预测
            outputs = model(img)
            # 设定阈值对框进行过滤
            for threshold in [0.1]:
                probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs,
                                                                           threshold=threshold,im=ori_img)
                # 将符合条件的框绘制到图片上
                plot_finetuned_results(ori_img, probas_to_keep, bboxes_scaled)
            ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

import time
import datetime
def img_predict(img_pth):
    global img_name
    # 读取模型网络
    # model, _, _ = build_model(args)
    # 也可以从GitHub上读取模型网络
    model = torch.hub.load('facebookresearch/detr',
                           'detr_resnet50',
                           pretrained=False,
                           num_classes=num_classes)

    # 读取自己训练好的模型权重
    checkpoint = torch.load('outputs/checkpoint.pth',
                            map_location='cpu')

    model.load_state_dict(checkpoint['model'],
                          strict=False)

    model.eval()

    # 读取图片进行预测
    img_name = img_pth
    im = Image.open(img_name)
    run_worflow(im, model)

img_name = ""

if __name__ == '__main__':
    # 预测图片

    img_predict(img_pth='/BOBO/datasets/detection_data/val2017/LZ63.jpg')

    # 获取训练的log
    # get_log_figure()

    # 预测视频
    # video_predict("E:/001.mp4")
