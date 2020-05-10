# 用于易盾图标点选的代码

开发于 python3

```
# 依赖 pytorch：（官网找安装方式，用一个比较新的版本即可）我开发使用版本为 torch-1.4.0-cp36-cp36m-win_amd64.whl
# 依赖 opencv： （pip install opencv-contrib-python==3.4.1.15）需要使用sift图像算法。所以注意安装版本。
    contrib 版简单理解成 opencv 的增强版就行。
```

内附少量样本直接执行即可测试，直接显示标注好的图片。代码稍加修改就能拿到自己想要的坐标信息了。

项目大小仅 3M。因为定位的 yolo 算法的网络被压缩到很小，便于下载。

脚本算法已做简单兼容，兼容 cpu和gpu两个版本。

完全准确率没有细测，用几十张图片肉眼测试，感觉上要比 50% 略高一点。