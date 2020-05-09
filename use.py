import cv2
import numpy as np
import torch
import torch.nn as nn

import os
import math

USE_CUDA = True if torch.cuda.is_available() else False
DEVICE = 'cuda' if USE_CUDA else 'cpu'

# 旋转图片的函数，目前没有被使用到
def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

# 将经过 backbone 的矩阵数据转换成坐标和分类名字
def parse_y_pred(ypred, anchors, class_types, islist=False, threshold=0.2, nms_threshold=0):
    ceillen = 5+len(class_types)
    sigmoid = lambda x:1/(1+math.exp(-x))
    infos = []
    for idx in range(len(anchors)):
        if USE_CUDA:
            a = ypred[:,:,:,4+idx*ceillen].cpu().detach().numpy()
        else:
            a = ypred[:,:,:,4+idx*ceillen].detach().numpy()
        for ii,i in enumerate(a[0]):
            for jj,j in enumerate(i):
                infos.append((ii,jj,idx,sigmoid(j)))
    infos = sorted(infos, key=lambda i:-i[3])
    def get_xyxy_clz_con(info):
        gap = 416/ypred.shape[1]
        x,y,idx,con = info
        gp = idx*ceillen
        contain = torch.sigmoid(ypred[0,x,y,gp+4])
        pred_xy = torch.sigmoid(ypred[0,x,y,gp+0:gp+2])
        pred_wh = ypred[0,x,y,gp+2:gp+4]
        pred_clz = ypred[0,x,y,gp+5:gp+5+len(class_types)]
        if USE_CUDA:
            pred_xy = pred_xy.cpu().detach().numpy()
            pred_wh = pred_wh.cpu().detach().numpy()
            pred_clz = pred_clz.cpu().detach().numpy()
        else:
            pred_xy = pred_xy.detach().numpy()
            pred_wh = pred_wh.detach().numpy()
            pred_clz = pred_clz.detach().numpy()
        exp = math.exp
        cx, cy = map(float, pred_xy)
        rx, ry = (cx + x)*gap, (cy + y)*gap
        rw, rh = map(float, pred_wh)
        rw, rh = exp(rw)*anchors[idx][0], exp(rh)*anchors[idx][1]
        clz_   = list(map(float, pred_clz))
        xx = rx - rw/2
        _x = rx + rw/2
        yy = ry - rh/2
        _y = ry + rh/2
        np.set_printoptions(precision=2, linewidth=200, suppress=True)
        if USE_CUDA:
            log_cons = torch.sigmoid(ypred[:,:,:,gp+4]).cpu().detach().numpy()
        else:
            log_cons = torch.sigmoid(ypred[:,:,:,gp+4]).detach().numpy()
        log_cons = np.transpose(log_cons, (0, 2, 1))
        for key in class_types:
            if clz_.index(max(clz_)) == class_types[key]:
                clz = key
                break
        return [xx, yy, _x, _y], clz, con, log_cons
    def nms(infos):
        if not infos: return infos
        def iou(xyxyA,xyxyB):
            ax1,ay1,ax2,ay2 = xyxyA
            bx1,by1,bx2,by2 = xyxyB
            minx, miny = max(ax1,bx1), max(ay1, by1)
            maxx, maxy = min(ax2,bx2), min(ay2, by2)
            intw, inth = max(maxx-minx, 0), max(maxy-miny, 0)
            areaA = (ax2-ax1)*(ay2-ay1)
            areaB = (bx2-bx1)*(by2-by1)
            areaI = intw*inth
            return areaI/(areaA+areaB-areaI)
        rets = []
        infos = infos[::-1]
        while infos:
            curr = infos.pop()
            if rets and any([iou(r[0], curr[0]) > nms_threshold for r in rets]):
                continue
            rets.append(curr)
        return rets
    if islist:
        v = [get_xyxy_clz_con(i) for i in infos if i[3] > threshold]
        if nms_threshold:
            return nms(v)
        else:
            return v
    else:
        return get_xyxy_clz_con(infos[0])

class Mini(nn.Module):
    class ConvBN(nn.Module):
        def __init__(self, cin, cout, kernel_size=3, stride=1, padding=None):
            super().__init__()
            padding   = (kernel_size - 1) // 2 if not padding else padding
            self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding, bias=False)
            self.bn   = nn.BatchNorm2d(cout, momentum=0.01)
            self.relu = nn.LeakyReLU(0.1, inplace=True)
        def forward(self, x): 
            return self.relu(self.bn(self.conv(x)))
    def __init__(self, anchors, class_types, inchennel=3):
        super().__init__()
        self.oceil = len(anchors)*(5+len(class_types))
        self.model = nn.Sequential(
            OrderedDict([
                ('ConvBN_0',  self.ConvBN(inchennel, 32)),
                ('Pool_0',    nn.MaxPool2d(2, 2)),
                ('ConvBN_1',  self.ConvBN(32, 48)),
                ('Pool_1',    nn.MaxPool2d(2, 2)),
                ('ConvBN_2',  self.ConvBN(48, 64)),
                ('Pool_2',    nn.MaxPool2d(2, 2)),
                ('ConvBN_3',  self.ConvBN(64, 80)),
                ('Pool_3',    nn.MaxPool2d(2, 2)),
                ('ConvBN_4',  self.ConvBN(80, 96)),
                ('Pool_4',    nn.MaxPool2d(2, 2)),
                ('ConvBN_5',  self.ConvBN(96, 102)),
                ('ConvEND',   nn.Conv2d(102, self.oceil, 1)),
            ])
        )
    def forward(self, x):
        return self.model(x).permute(0,2,3,1)

def get_clz_rect(filename, state):
    net = state['net'].to(DEVICE)
    optimizer = state['optimizer']
    anchors = state['anchors']
    class_types = state['class_types']
    net.eval() # 重点中的重点，被坑了一整天。
    npimg = cv2.imread(filename)
    height, width = npimg.shape[:2]
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB) # [y,x,c]
    npimg = cv2.resize(npimg, (416, 416))
    npimg_ = np.transpose(npimg, (2,1,0)) # [c,x,y]
    y_pred = net(torch.FloatTensor(npimg_).unsqueeze(0).to(DEVICE))
    img = cv2.imread(filename)
    v = parse_y_pred(y_pred, anchors, class_types, islist=True, threshold=0.2, nms_threshold=0.4)
    ret = []
    for i in v:
        rect, clz, con, log_cons = i
        rw, rh = width/416, height/416
        rect[0],rect[2] = int(rect[0]*rw),int(rect[2]*rw)
        rect[1],rect[3] = int(rect[1]*rh),int(rect[3]*rh)
        ret.append([clz, rect])
    return ret

def get_cut_img(npimg, rects):
    ret = []
    for clz, (x1,y1,x2,y2) in rects:
        ret.append([clz, npimg[y1:y2,x1:x2,:], (x1,y1,x2,y2)])
    return ret


# 处理顺序的问题
def get_flags_rects(file, state):
    s = cv2.imread(file)
    a = s[160:,0*28:1*28-6,:]
    b = s[160:,1*28:2*28-6,:]
    c = s[160:,2*28:3*28-6,:]
    a1, a2 = a[40:60], a[0:20]
    b1, b2 = b[40:60], b[0:20]
    c1, c2 = c[40:60], c[0:20]
    def get_match_lens(i1, i2):
        i1 = cv2.resize(i1, (int(i1.shape[1]*8), int(i1.shape[0]*8)))
        i2 = cv2.resize(i2, (i2.shape[1]*4, i2.shape[0]*4))
        s = cv2.xfeatures2d.SIFT_create()
        kp1,des1 = s.detectAndCompute(i1,None)
        kp2,des2 = s.detectAndCompute(i2,None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        DIS = .88
        for m,n in matches:
            if m.distance <= DIS * n.distance:
                good.append([m])
        i3 = cv2.drawMatchesKnn(i1,kp1,i2,kp2,good,None)
        # cv2.imshow('nier', i3)
        # cv2.waitKey(0)
        return len(good)
    def get_flag_rect(k12, cut_imgs, st):
        k1, k2 = k12
        r = []
        for clz, npimg, rect in cut_imgs:
            if clz == '1':
                r1 = get_match_lens(k1, npimg)
                r.append([r1, rect, st])
            if clz == '2':
                r2 = get_match_lens(k2, npimg)
                r.append([r2, rect, st])
        return sorted(r, key=lambda i:i[0])
    v = get_cut_img(s, get_clz_rect(file, state))
    rs1 = get_flag_rect([a1, a2], v, 1)
    rs2 = get_flag_rect([b1, b2], v, 2)
    rs3 = get_flag_rect([c1, c2], v, 3)
    rs = rs1 + rs2 + rs3
    r = []
    t = []
    v = max([j for j in rs if j[2] not in t], key=lambda i:i[0])
    r.append(v)
    t.append(v[2])
    q = []
    for i in rs:
        if i[1] == v[1]:
            q.append(i)
    for i in q:
        rs.remove(i)
    v = max([j for j in rs if j[2] not in t], key=lambda i:i[0])
    r.append(v)
    t.append(v[2])
    q = []
    for i in rs:
        if i[1] == v[1]:
            q.append(i)
    for i in q:
        rs.remove(i)
    v = max([j for j in rs if j[2] not in t], key=lambda i:i[0])
    r.append(v)
    t.append(v[2])
    r1, r2, r3 = sorted(r,key=lambda i:i[2])
    return r1[1], r2[1], r3[1]

def draw_rects(filename, rects):
    def drawrect(img, rect, text):
        cv2.rectangle(img, tuple(rect[:2]), tuple(rect[2:]), (10,250,10), 2, 1)
        x, y = rect[:2]
        cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10,10,250), 1)
        return img
    img = cv2.imread(filename)
    for idx, rect in enumerate(rects, 1):
        img = drawrect(img, rect, '{}'.format(idx))
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

xmlpath = './testimg'
v = [os.path.join(xmlpath, i) for i in os.listdir(xmlpath) if i.endswith('.jpg')]
v = v[::-1]

print('loading net')
state = torch.load('net.pkl')
print('loading net ok.')
for file in v:
    print(file)
    rects = get_flags_rects(file, state)
    draw_rects(file, rects)