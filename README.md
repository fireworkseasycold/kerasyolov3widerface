# YOLO v3 目标检测算法 源码

> 欢迎关注，微信公众号 **深度算法** （ID: DeepAlgorithm） 

相关文章：

- [探索 YOLO v3 源码 - 第1篇 训练](https://mp.weixin.qq.com/s/T9LshbXoervdJDBuP564dQ)
- [探索 YOLO v3 源码 - 第2篇 模型](https://mp.weixin.qq.com/s/N79S9Qf1OgKsQ0VU5QvuHg)
- [探索 YOLO v3 源码 - 第3篇 网络](https://mp.weixin.qq.com/s/hC4P7iRGv5JSvvPe-ri_8g)
- [探索 YOLO v3 源码 - 第4篇 真值](https://mp.weixin.qq.com/s/5Sj7QadfVvx-5W9Cr4d3Yw)
- [探索 YOLO v3 源码 - 第5篇 Loss](https://mp.weixin.qq.com/s/4L9E4WGSh0hzlD303036bQ)
- [探索 YOLO v3 源码 - 完结篇 预测](https://mp.weixin.qq.com/s/J1ddmUvT_F2HcljLtg_uWQ)

通过6篇文章，完整的呈现YOLO v3的源码细节。慢慢读完，掌握一些高级的深度学习开发技巧。

参考：

- [YOLO v3 Paper](https://arxiv.org/abs/1804.02767)
- [What’s new in YOLO v3?](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)

勘误：

1. 第4篇 真值，最后：“y_true的第0和1位是中心点xy，范围是`(0~13/26/52)`” -> “y_true的第0和1位是中心点xy，范围是`(0~1)`”；
2. 第3篇 网络，其中关于补充部分``1*1``卷积参数那个有误。不是``13*13*1*1*18``应该是``1*1*1024*18``； Thx@草绛ly
3. 第6篇 预测，max_boxes是在每层的feature_map中的每个类别分别最多产生20个框，而不是每张图片； Thx@略略略


##以下是我的记录：
#author：fireworkseasycold,可以followme
##环境：
#显卡gtx1050ti-4g，系统win102004，msi笔记本，8g内存
#下载cuda9.2.148(针对gtx1050ti),
#下载对应版本cudnn，cudnn9.2v7.6.5.32
#环境管理工具anconda3，可根据需要配置清华源，豆瓣，中科院等源
#创建python=3.6.2虚拟环境，并activate此环境(也许3.6.10也行，反正不要3.6.0，一般会有问题)
# 查看可获得的tensorflow-gpu版本 conda search --full --name tensorflow-gpu
conda install tensorflow-gpu=1.12.0 #对应cuda9系列，后续我装了10.0还能用，也许是因为虚拟环境下安装的cudatoolkit=9.2?
#注意conda安装1.12安装自带cudatoolkit=9.0和cudnn与显卡推荐的不一致,我这里测试tensorflow会报错
#cuda9.2更新驱动为451.67，驱动无法安装注意升级windows10至2004，然后此问题解决
conda install keras #查看keras版本print(keras.__version__)
#import keras；print(keras.__version__)报错AttributeError: module 'tensorflow.python.keras.backend' has no attribute 'get_graph'
#解决方法，conda uninstall keras,然后到此网站查看tensorflow1.12对应keras版本应该是2.2.4，https://docs.floydhub.com/guides/environments/
#tensorbard，如果启动报protbuf错，改成3.7.1，还报错就改成3.6.1
conda install pillow
conda install matplotlib
#graphviz==0.8.4和pydot==1.2.4我没装，所以我把yolov3_train.py里plot_model这一行注释了，这是keras的可视化
#sklearn,seaborn根据需要自己装，训练用不上
##训练：
#数据：根据标注图片用wider_annotation.py生成需要的configs/WIDER_train.txt，我这里提供了
#你的类别wider_classes.txt
#yolo_anchors.txt我这没改，你可以用kmeans.py改
#预训练模型可以加载也可改成False,可使用官方yolov3.h5(由yolov3.weights转生成,自己百度个脚本，200多兆)，或者上个结点
#注意爆显存：改小输入416*416，我这改成64*64才不爆
#改好路径执行train
#只改这么多大概一晚上，显存利用率低，可以改写
##可视化
tensorboard --logdir=yolov3_train里的log_dir，打开localhost:6006
#示例：show
##上传至github或者gitlab仓库：
#创建GitHub仓库,啥都不选，否则可能冲突
##初始化本地仓库
git init
#添加暂存
git add *或. 
#"."代表这个test这个文件夹下的目录全部都提交，或者改为你要提交的文件名字
#查看下现在的状态
git status
#把文件提交的本地仓库
git commit -m "这里面写你的注释"
##忽略项
#1、在需要创建 .gitignore 文件的文件夹, 右键选择Git Bash 进入命令行，进入项目所在目录。
#2、输入 touch .gitignore ，生成“.gitignore”文件。
#3、在”.gitignore” 文件里输入你要忽略的文件夹及其文件就可以了。（注意格式）