## R-CNN

### Pre knowledge

参考[RCNN- 将CNN引入目标检测的开山之作](https://zhuanlan.zhihu.com/p/23006190?refer=xiaoleimlnote)。

1. 重叠度（IOU）

   Bounding box是标注物体的框体，IOU就是ground truth bounding box 与 Inference bounding box的重叠面积占二者并集的面积比例。

   ![IOU = (A∩B) / (A∪B)](https://pic1.zhimg.com/80/v2-6fe13f10a9cb286f06aa1e3e2a2b29bc_hd.jpg)

2. 非极大值抑制（NMS）

   算法会从一张图片中找出n个可能是物体的矩形框，需要判断哪些有用。算法步骤如下：

   + 对所有框按分类是否**属于指定类的概率**从小到大排序。
   + 从**最大概率**的矩形框开始，分别判断n - 1个框与最大概率框的**重叠度IOU**是否**大于某个设定的阈值**。超过阈值的框除去，保留最大阈值的框并标记。
   + 从剩下的框中再选取概率最大的框，然后重复第二步。一直到找到所有被保留下来的矩形框。

3. mAP：mean average precision（均值平均精度）。在目标检测中衡量识别精度。多个类别物体检测中，每一个类别都可以根据recall和precision绘制一条曲线。AP（average precision）就是该曲线下的面积，mAP是多个类别AP的平均值。
   $$
   MeanAveragePrecision = \frac {\Sigma AveragePrecision_C} {N(Classes)}
   $$
   其中
   $$
   AveragePrecision_C = \frac {\Sigma Precision_C} {N(TotalImages)_C}
   $$

   $$
   Precision_C = \frac {N(TruePostivies)_C} {N(TotalObjects)_C}
   $$








### 步骤

1. 候选区域生成：一张图像生成1K-2K个候选区域（使用Select Search方法）
   + Select Search 方法：
     1. 首先基于[Efficient GraphBased Image Segmentation](http://cs.brown.edu/~pff/segment/)（[笔记 - 基于图的图像分割](https://blog.csdn.net/ttransposition/article/details/38024557)）方法得到region。
     2. 使用贪心策略，计算每两个相邻的区域的**相似度**，每次合并最相似的两块，直至最终只剩下一张完整的图片。其中每次产生的图像块（包括合并的图像块都保存下来），即可得到图像分层表示。
        + 如何计算两个区域的相似度？需要计算颜色距离（颜色直方图相似）、纹理距离（梯度直方图相似）、给与小区域较高的合并优先级、两个区域的吻合度等，通过加权综合各种距离（详见[Select Search](https://zhuanlan.zhihu.com/p/27467369)）。
     3. 通过以上步骤可以得到很多区域，不同区域作为目标的可能性不同，需要给区域**打分**。给**最先合并的图片块**较大的权重，最后一块完整图像权重为1，倒数第二次合并的权重为2，以此类推。同时权重再乘以一个随机数，按最后得到的结果进行**排序**。
     4. 根据需要选取分数最高的N个候选区域。
2. 特征提取：对每个候选区域，使用**CNN**提取特征。
   + 输入是227 * 227图像，由于selective search得到的是矩形框，paper使用了各向异性缩放（即不管图片长宽比例，缩放即可）。原文中做了padding = 16处理。
   + 使用CNN从每个候选区域中提取一个**固定长度的特征向量（4096维）**。
   + CNN使用AlexNet或VGG16。举例使用AlexNet，参数也使用AlexNet预训练的参数。
   + Fine-tuning训练：假设要检测的物体有N类，则将预训练的CNN模型最后一层替换掉，换成N + 1个输出的神经元（N个类型 + 1个背景类型）。batch size大小为128，其中32个正样本，96个负样本。
     + 正/负样本判断：Select search挑选出来的候选框与ground truth物体框的IOU > 0.5时，则标记为正样本（没有具体类别，只是告诉SVM分类器这里是目标样本，这就是**region proposal**），否则标记为背景类别（负样本）。
3. 类别判断：特征送入每一类的SVM分类器，判断是否属于该类。
   + 将**CNN中输出为正样本**的**候选框**作为SVM分类器的输入。定义候选框与ground truth的IOU**小于0.3**时，标为负样本（背景样本）。
   + 为**每个物体类**训练一个svm分类器。假设我们用CNN提取2000个候选框，可以得到2000 * 4096这样的特征向量矩阵，然后我们只需要把这样的一个矩阵与svm权值矩阵4096 * N点乘(N为分类类别数目，因为我们训练的N个svm，每个svm包含了4096个权值w)，即可得到分类结果。
   + 使用**非极大值抑制（NMS）**去除多余的框，排序，canny边界检测之后得到bounding-box。
4. 位置精修：使用回归器修正候选框的位置。
   + 对每一类目标，使用一个**线性脊回归器**进行精修。
   + 正则项λ=10000。 
   + 输入为深度网络pool5层的4096维特征，输出为xy方向的**缩放**和**平移**。 
   + 训练样本：**判定为本类的候选框中**和ground truth重叠面积**大于0.6**的候选框。

---

## SPP-Net

出自论文《Spatial Pyramid Pooling in Deep ConvolutionalNetworks for Visual Recognition 》。

R-CNN存在一些性能瓶颈：

+ 速度瓶颈。Selective Search对每张图像产生2k个region proposal，意味着一幅图片需要经过2k次完整的CNN才能得到最终结果。
+ 性能瓶颈。由于输入使用了各向异性缩放，会导致图像**几何畸变**。但由于全连接层中需要固定的输入尺寸，不能使用任意尺度的未经缩放的图像。

###  解决全连接层处理不同尺寸输入的思路

使用**SPM（Spaital Pyramid Matching）**方式。将一副图像分成若干尺度的一些块，比如将一副图像分成1份，4份，8份等，然后将每一块特征融合到一起，就可以得到多个尺度的特征。

![空间金字塔池化层](https://pic1.zhimg.com/v2-62c008799df798656236258c64082340_r.jpg)

上图的空间金字塔池化层是SPPNet的核心，其主要目的是对于任意尺寸的输入产生固定大小的输出。思路是对于任意大小的feature map首先分成16、4、1个块，然后在每个块上最大池化，池化后的特征拼接得到一个固定维度的输出。以满足全连接层的需要所以输入的图像为**一整副图像**。

### 多尺度训练

为了能够**更快地收敛**，paper中采用了**多尺度训练**的方法：使用两个尺寸（224 * 224和180 * 180）的图像训练。180 * 180的图像通过224 * 224缩放得到。之后交替训练，用224的图像训练一个epoch，用180的图像训练一个epoch。

这里使用不同size的图像训练时，全连接层之前的最后一层金字塔池化层中，window_size和stride_size这两个参数由输入图像的size决定。

令conv5出来后特征图的size为a * a，金字塔池化层的size为n * n。则

window_size = [a / n]向上取整。

stride_sizes = [a / n]向下取整。

例如对于pool 3 * 3（这里就不是指的是池化层本身size，而是输出时pool的小块的个数），input image size = 224 * 224，经过5层卷积后feature map为 13 * 13，有

widnows_size = [13 / 3]向上取整 = 5

stride = [13 / 3]向下取整 = 4

同理，当input image size = 180 * 180，经过5层卷积后feature map为 10 * 10，有

widnows_size = [10 / 3]向上取整 = 4

stride = [10 / 3]向下取整 = 3

这样在两种尺度下的SSP后，输出特征维度都是(9 + 4 + 1) * 256，256是feature map个数（channels）。参数是共享的，之后连接全连接层即可。

### Mapping a Window to Feature Maps

在原图中的proposal,经过多层卷积之后，位置还是相对于原图不变的（如下图所示），那现在需要解决的问题就是，如何能够将原图上的proposal,映射到卷积之后得到的特征图上，因为在此之后我们要对proposal进行金字塔池化。

![卷积后特征图中特征位置](https://pic3.zhimg.com/80/v2-523707e94ccb850ca4c23cc94054a144_hd.jpg)

假设$(x, y)$是输入图像上的坐标，$(x', y')$是feature map上的坐标，则映射关系如下：

左上角：$x' = [x / S] + 1$，x / S向下取整

右下角：$x' = [x / S] - 1$，x / S向上取整

其中S为之前**所有层stride的乘积**。

### 分类

最后得到的feature map中的proposal特征框，与在R-CNN中一样，送入SVM中进行分类。

### 存在的不足

和RCNN一样，SPP也需要训练CNN提取特征，然后训练SVM分类这些特征。需要巨大的存储空间，并且分开训练也很复杂。而且selective search的方法提取特征是在CPU上进行的，相对于GPU来说还是比较慢的。

---

## Fast R-CNN

+ 主要对R-CNN中使用大量重叠的region proposal进行卷积造成的**提取特征操作冗余**进行了改进
+ 同时把类别判断和位置精调**统一用深度网络实现**，不再需要额外存储

### 流程图

![FAST R-CNN流程图](https://pic2.zhimg.com/v2-9f58e8489c22b7a5809feac4f743491f_r.jpg)

### ROI Pooling

与SPP的目的相同，将不同尺寸的ROI映射为固定大小的特征，不同的是ROI Pooling没有考虑多个空间尺度，只使用单个尺度。

ROI pool层将每个region proposal均匀分为M * N块，对每一块进行max pooling，从而将feature map上大小不一的region proposal转变为大小统一的数据，送入下一层。对每一个region proposal的pooling网格大小都要单独计算。（参考[FAST R-CNN](https://zhuanlan.zhihu.com/p/24780395)）。

![ROI Pooling](http://pb2ofoe75.bkt.clouddn.com/ROIPooling.jpg)

### Bounding-box Regression

FAST R-CNN去掉了SVM分类层与线性回归精修候选框层，将最后一层的**softmax层换成两个**，一个是对region的分类（包括背景），另一个是对bounding box进行微调。

论文在SVM和Softmax的对比实验中说明，SVM的优势并不明显，故直接用Softmax将整个网络整合训练更好。对于联合训练，同时利用了**分类的监督信息和回归的监督信息**，使得网络训练的更加鲁棒，效果更好。这两种信息是可以有效联合的。

在位置精修中，使用的是**smooth损失函数**。假设对于类别$k^*$，在图像中标注了一个ground truth坐标$t^* = (t^*_x, t^*_y, t^*_w, t^*_h)$，即锚点坐标、宽、高。预测值为$t = (t_x, t_y, t_w, t_h)$。定义损失函数
$$
L_{loc}(t, t^*) = \Sigma_{i \in \{x, y, w, h\}}smooth_{L1}(t_i, t^*_i)
$$
其中
$$
smooth_{L1}(x) = \begin{cases} 0.5x^2, &|x|\le1 \\ |x| - 0.5 &otherwise  \end{cases}
$$
其中x为$t_i - t_i^*$，即对应坐标与高宽的差距，该函数在(-1, 1)之间为二次函数，其他区域为线性函数。

![smooth函数](https://pic2.zhimg.com/80/v2-fa78a0462cb6cd1cd8dcd91f88820d19_hd.jpg)

详细的regression过程参考[FAST R-CNN](https://zhuanlan.zhihu.com/p/24780395)。

---

## Faster R-CNN

整体结构如下。

![Faster R-CNN](http://pb2ofoe75.bkt.clouddn.com/Faster%20R-CNN.jpg)

一个网络结构，包含了4个损失函数：

+ RPN classification（anchor -> foreground / background）
+ RPN regression（anchor -> proposal）
+ Faster R-CNN classification（2000 classes）
+ Faster R-CNN regression（proposal ->  bounding box）

### Region Proposal Network（区域候选网络）

训练一个网络来替代selective search，进行物体的框选。

![RPN](http://pb2ofoe75.bkt.clouddn.com/RPN.jpg)

RPN网络分为两个支路，上面一条通过softmax二分类anchors获得foreground（检测目标）和background，下面一条用于计算对于anchors的bounding box regression的偏移量，以获得精确的proposal。最后的Proposal层则负责综合foreground anchors和bounding box regression偏移量获得proposals。同时剔除太小和超出边界的proposals。

#### Anchors

Anchor实际上就是一组有编号有坐标的bounding box，一共**3种形状，9个矩形（3种尺度）**，即长宽比为width : height = [1:1, 1:2, 2:1]。

![Anchors](http://pb2ofoe75.bkt.clouddn.com/Anchors.jpg)

RPN之前的卷积操作完成后，遍历得到的feature maps，为每一个点使用以上9种anchors作为初始检测框，后面还会修正检测框位置。得到所有的anchor boxes后，再使用**NMS（非极大值机制，见R-CNN中的详细介绍）**移除部分候选框，然后进行训练。

![Anchors处理过程](http://pb2ofoe75.bkt.clouddn.com/Anchors%E5%A4%84%E7%90%86%E8%BF%87%E7%A8%8B.png)

#### Bounding box regression原理

详见[ CNN目标检测（一）：Faster RCNN详解](https://www.jianshu.com/p/de37451a0a77)第2.4节，或者[Faster R-CNN](https://zhuanlan.zhihu.com/p/24916624)。原理与Fast R-CNN中基本相同，只是$t_i^*$由原来selection search的候选框变成了anchor box。

### 后续步骤

经过RPN训练后得到的proposal，与featrue maps进行concat，然后进行POI Pooling，使Fully Connection层的输入完全一致。然后进行一个Multi-task的训练任务，得到结果。

---

## YOLO

YOLO（You Only Look Once）将物体检测作为**回归问题**求解，基于一个单独的end-to-end网络，完成从原始图像到物体位置和类别的输出。输入图像经过一次inference，就能得到图像中所有物体的位置和其所属的类别及相应的置信概率（confidence）。

### 步骤

1. 将图像resize成448 * 448，图像分割为7 * 7网格（cell），这里分割的数量可以任意，后面计算时按照分割的size计算。
2. CNN提取特征和预测：
   + 7 * 7 * 2bounding box（bbox）的坐标$(x_{center}, y_{center}, w, h)$，和7 * 7 * 2个是否有物体的confidence。
   + 7 * 7个cell分别属于哪一个物体的概率（paper中为20种物体）。
3. 通过NMS过滤bbox。

### 网络结构

![YOLO](https://pic1.zhimg.com/80/v2-2c4e8576b987236de47f91ad594bf36d_hd.jpg)

YOLO包含了24个卷积层和2个全连接层，其借鉴了GoogleNet的结构，但未使用Inception方式，而是使用了1 * 1卷积层 + 3 * 3卷积层替代。

### 训练

#### 预训练分类网络

在ImageNet 1000-class competition dataset上预训练一个分类网络，这个网络是网络结构中前20个卷积网络 + average-pooling layer + fully connection layer（此时网络的输入是**224 * 224**）。

#### 训练检测网络

这里添加了4个卷积层和2个全连接层，将网络输入改为**448 * 448**。

+ 一副图片分成7 * 7个grid cell，当物体的ground truth bbox的中心落在某个网络中时，该网络就负责预测这个物体（是回归检测框还是分类？）。例如以下这张图，狗的中心点落在(5, 2)这个格子内，这个格子就负责预测图像中的物体狗。

  ![物体中心点](https://pic2.zhimg.com/80/v2-b646262ae5e3cf7d55a2f774d49d61c0_hd.jpg)

+ 每个格子输出B个bounding box，Paper中**B = 2**，如下图中的黄色框。这里的bounding box是**人为选定的2个不同长宽比的box**。也就是说每个cell要预测两个bounding box（四个坐标信息$(x_{center}, y_{center}, w, h)$和一个confidence值）。其中：

  + 中心坐标$x_{center}, y_{center}$相对于对应的网格归一化到0 - 1之间，w, h用图像的width和height归一化到0-1之间。

  + confidence代表了所预测的bbox中含有object的置信度和这个bbox预测的有多准的两重信息：
    $$
    confidence = Pr(Object) * IOU^{truth}_{pred}
    $$
    如果有ground truth box落在grid cell里（不一定是bbox的中心，只要是box和grid cell有交集就算），第一项取1，否则取0。第二项是预测bounding box和实际的ground truth之间的IOU值。

![bounding box](https://pic2.zhimg.com/v2-1ad557fda288473b0335fe64e03bc049_r.jpg)

+ 预测类别信息。Paper中有20类。

因此，每一个网格要预测2个bounding box（2 * （$(x_{center}, y_{center}, w, h)$ + confidence））和20类，一共30个参数。因此输出tensor的维度为 **7 * 7 * 30 = 1470**。

### Loss Function定义

YOLO的loss function基本思路如下：
$$
loss = \Sigma^{S^2}_{i=0}(coordError + IOUError + classError)
$$
同时，对以上公式进行如下修正：

 + 位置相关误差（coordError和IOUError）与分类误差（calssError）对网络loss的贡献值是不同的，使用$\lambda_{coord}=5$修正coordError。
 + 在计算IOU误差时，包含物体的cell和不包含物体的cell，两者的IOUError对loss的贡献值是不同的，若采用相同的权值，那么不包含物体的格子的confidence值近似为0，变相放大了包含物体的格子的confidence误差在计算网络参数梯度时的影响。YOLO使用$\lambda_{noobj} = 0.5$修正IOUError。（包含指的是**物体grounth truth box的中心坐标落到格子内**）。
 + 对于相等的误差值，大物体误差对检测的影响应小于小物体误差对检测的影响。这是因为，**相同的位置偏差占大物体的比例远小于同等偏差占小物体的比例**。YOLO将物体大小的信息项（w和h）进行求**平方根**来改进这个问题。

修正后的loss function如下：

![loss function](https://pic2.zhimg.com/v2-c629e12fb112f0e3c36b0e5dca60103a_r.jpg)

其中, x, y, w, C, p为label值，x, y, w, C, p hat为预测值。$\Pi ^{obj}_i$表示**物体落入格子i**中，$\Pi ^{obj}_{ij}$表示物体**落入格子i的第j个bounding box**, $\Pi ^{noobj}_{ij}$表示物体**没有落入格子i的第j个bounding box**。

### Inference

对图像中的7 * 7 * 2个bounding box进行前向，计算confidence。设置阈值过滤掉得分低的boxes。然后对保留的boxes进行NMS处理，得到最终的检测结果。

### 缺陷

+ YOLO对相互靠近的物体、很小的群体检测效果不好。
+ 当同一类物体出现不常见的长宽比时，泛化能力较弱。
+ Loss Function中对不同大小相同误差值的物体处理不能完全解决问题（只是一个trick），影响定位检测效果。

---

## SSD

### 网络结构

SSD（Single Shot MultiBox Detector）采用了VGG16的基础网络结构，使用前面的5层，然后利用atrous算法将fc6和fc7层转化成两个卷积层，再额外增加3个卷积层和1个average pool层。不同层次的feature map分别用于default box的偏移以及不同类别得分的预测，最后通过NMS得到最终结果。

![SSD网络结构和与YOLO的比较](https://pic4.zhimg.com/80/v2-5d36659e8be837ad165b0c10210b95af_hd.jpg)

增加的卷积层的feature map大小变化较大，允许能够检测出不同尺度下的物体。SSD中每一层的输出只会感受到目标周围的信息，使用不同的feature map和不同的default box预测不同宽高比的图像，比YOLO增加了预测更多比例的box。

![SSD预测bbox示例](https://pic3.zhimg.com/v2-5964f6dff6dbbd435336cde9e5dfc988_r.jpg)

#### Atrous卷积

Atrous卷积就是带洞卷积，卷积核是稀疏的。例如下图中的第三个示例（c），就是带洞卷积。带洞卷积的有效性基于一个假设：**紧密相邻的像素几乎相同，全部纳入属于冗余，不如跳H个(H为hole size)取1个**。

![Atrous卷积](https://pic1.zhimg.com/80/v2-0d79b7626e95e1abad89d7f3d1707088_hd.jpg)

### 先验框

SSD借鉴了Faster R-CNN中anchor的理念，每个单元设置尺度或者长宽比不同的先验框。原文中的图如下。图中框出的是每一个点有4个先验框（原文中并没有给出具体的先验框个数，而是给了一个变量k个，有的博客上说是k = 6）。

![SSD的先验框](https://pic1.zhimg.com/80/v2-f6563d6d5a6cf6caf037e6d5c60b7910_hd.jpg)

综上所述，对一个大小为 m * n的特征图，每一个单元设置先验框数目为k，分类的数量为c（包括1个背景类），default box偏移量为4（即$x, y, w, h$），**所有单元共需要$(c+4) * k * m * n$个预测值**。

### 训练策略

#### 正负样本

找到每个物体ground true box对应的default box中**IOU最大**的作为正样本，再在剩下的default box中找到那些与任意一个ground truth box的IOU大于0.5的default box作为正样本。即**一个ground truth可能对应多个正样本default box**。其他的作为负样本。

尽管一个ground truth可以与多个先验框匹配，但是ground truth相对先验框还是太少了，所以负样本相对正样本会很多。为了保证正负样本尽量平衡，SSD采用了**hard negative mining**，就是对负样本进行抽样，抽样时按照**置信度误差（预测背景的置信度越小，误差越大）进行降序排列**，选取误差的较大的top-k作为训练的负样本，以保证正负样本比例接近**1:3**。

#### 损失函数

损失函数定义为位置误差（loc）与置信度误差（conf）的加权和：
$$
L(x, c, l, g) = \frac {1} {N} (L_{conf}(x, c) + \alpha L_{loc}(x, l, g))
$$
其中N是先验框的正样本数量，这里$x^p_{ij} \in \{1, 0\}$是一个指示参数，当$x^p_{ij} = 1$时表示第i个先验框与第j个ground truth匹配，并且ground truth类别为p。c为类别置信度预测值。l为先验框的对应边界框的位置预测值。g是ground truth的位置参数。

对于位置误差，采用Smooth L1 loss，定义如下：

![smooth L1 loss](https://pic1.zhimg.com/80/v2-a56019049a04217560ac498b626ad916_hd.jpg)

![smooth](https://pic4.zhimg.com/80/v2-5853bed6e8796de582647f1e557b623b_hd.jpg)

对于置信度误差，采用softmax loss：

![置信度误差](https://pic2.zhimg.com/80/v2-d28ded21949483b0fbb64b3612b0d543_hd.jpg)

权重系数$\alpha$通过交叉验证设置为1。

### 预测过程

对于每个预测框，首先根据类别置信度确定其类别（置信度最大者）与置信度值，并过滤掉属于背景的预测框。然后根据置信度阈值（如0.5）过滤掉阈值较低的预测框。对于留下的预测框进行解码，根据先验框得到其真实的位置参数（解码后一般还需要做clip，防止预测框位置超出图片）。解码之后，一般需要根据置信度进行降序排列，然后仅保留top-k（如400）个预测框。最后就是进行NMS算法，过滤掉那些重叠度较大的预测框。最后剩余的预测框就是检测结果了。

---

## YOLOv2

YOLOv2在原论文中称为YOLO9000，同时在COCO和ImageNet数据集中进行训练，训练后模型可以实现多达9000种物体的实时检测，VOC 2007数据集测试，67FPS下mAP达到76.8%，40FPS下mAP达到78.6%。原文标题种提到了YOLOv2相比于v1，Better，Faster and Stronger。

### Better

#### Batch Normalization

YOLOv2在每一个卷积层后添加BN层，极大改善了收敛速度同时减少了对其他regularization方法的依赖，使mAP获得了2%的提升。

#### High Resolution Classifier

YOLOv2预训练分类网络的分辨率为448 * 448，在ImageNet数据集上训练10个epochs。再fine tune训练整个网络，mAP提升了4%。

#### Convolutional With Anchor Boxes

YOLOv2借鉴了Faster R-CNN中的anchor思想，为卷积特征图进行滑窗采样，每个中心预测9种不同大小和比例的proposal box。使用预测offset取代直接预测坐标简化了问题，方便网络学习。

具体所有的fine tune过程详见[YOLO2](https://zhuanlan.zhihu.com/p/25167153)。

#### Dimension Clusters（维度聚类）

使用anchor时，作者发现Faster-RCNN中anchor boxes的个数和宽高维度往往是手动精选的先验框（hand-picked priors)，设想能否一开始就选择了更好的、更有代表性的先验boxes维度，那么网络就应该更容易学到准确的预测位置。解决办法就是统计学习中的**K-means聚类方法**，通过对数据集中的**ground true box做聚类**，找到ground true box的统计规律。以**聚类个数k为anchor boxs个数**，以**k个聚类中心box的宽高维度为anchor box的维度**。

如果按照k-means使用欧式距离函数，大boxes比小boxes产生更多error。但是，我们真正想要的是产生好的IOU得分的boxes（与box的大小无关）。因此采用了如下距离度量：
$$
d(box, centroid) = 1 - IOU(box, centroid)
$$
聚类结果如图：

![维度聚类结果](https://pic1.zhimg.com/80/v2-e2fd3e6743a9598367093fa2836f0db8_hd.jpg)

上面左图： 随着k的增大，IOU也在增大（高召回率），但是复杂度也在增加。所以平衡复杂度和IOU之后，最终得到**k值为5**。

上面右图：5聚类的中心与手动精选的boxes是完全不同的，扁长的框较少瘦高的框较多。

作者做了对比实验，5种boxes的Avg IOU(61.0)就和Faster R-CNN的9种Avg IOU(60.9)相当。 说明K-means方法的生成的boxes更具有代表性，使得检测任务更好学习。

#### Direct location prediction

Paper中不再预测anchor box的中心点与高宽，而是预测**相对于grid cell的坐标位置**，同时把ground truth限制在0-1之间（使用logistic激活函数约束网络的预测值（其实是sigmoid？））。在RPN中，预测(x, y)以及$t_x, t_y$使用的是如下公式：
$$
x = (t_x * w_a) - x_a \\
y = (t_y * h_a) - y_a
$$
最终，网络在feature map(13 * 13)的每个cell上预测5个bounding boxes，每一个bounding box预测5个坐标值：$t_x, t_y, t_w, t_h, t_o$。如果这个cell距离图像左上角的边距为$(c_x, c_y)$以及该cell对应box的长和宽分别为$(p_w, p_h)$，那么预测值可以表示为：
$$
b_x = \sigma(t_x) + c_x \\
b_y = \sigma(t_y) + c_y \\
b_w = p_we^{t_w} \\
b_h = p_he^{t_h} \\
Pr(object) * IOU(b, object) = \sigma(t_o)
$$
$\sigma$函数是把归一化值转化为图中真实值，使用e的幂函数是因为前面做了ln计算。详见[YOLOv2论文笔记](https://blog.csdn.net/jesse_mx/article/details/53925356)。

![直接位置预测](https://pic3.zhimg.com/80/v2-dafc6f95636611bfcda7244738418e32_hd.jpg)

---

## Mask R-CNN

### 总体框架

![Mask R-CNN框架](http://pb2ofoe75.bkt.clouddn.com/Mask_R_CNN.jpg)

上图是Mask R-CNN的框架。其中黑色部分是原来的Faster R-CNN，红色部分为作者在Mask R-CNN中做的改进。主要为以下两点：

+ 将ROI Pooling层替换成了**ROI Align**。
+ 添加与ROI Align并列的**FCN（Fully Convolutional Network）层**，也就是分割mask的分支。通过应用到每一个ROI上的一个小FCN，以pix2pix的方式预测分割mask。

### Mask R-CNN中对Faster R-CNN的改进

#### ROI Align

之所以提出ROI Align，是因为之前的ROI Pooling操作会让实例分割的结果会有较大的重叠，而ROI Align很好地解决了ROI Pooling操作中两次量化造成的**miss-alignment**问题。

由于 Fast R-CNN 、Faster R-CNN 的候选框位置是通过 regression 方式得到，值大部分是**浮点数**，而要执行完成 RoIPooling 有需要的得到**固定尺寸的特征映射**。因此 RoIPooling 存在两次量化的过程：

+ 将候选框边界量化为整数点坐标值。
+ 将量化后的边界区域平均分割成 k x k 个单元(bin),对每一个单元的边界进行量化（如下图）。

![ROI Pooling过程](https://pic4.zhimg.com/v2-e55e66c9c21136efa62beacfcb02f9ef_b.gif)

这样经过两次量化之后，得到的候选框和最初的regression得到的位置就出现了一定偏差，即miss-alingment。这对分类没有影响，但对于预测mask有很大影响。因此，为了解决该问题，作者提出了ROI Align。

ROI Align通过取消量化操作，使用**双线性内插值（bilinar interpolation）**方法获得坐标为浮点数的像素点上的数值。将整个特征聚集的过程转化为一个连续的操作。操作流程如下（具体案例见[Mask R-CNN阅读记录](https://zhuanlan.zhihu.com/p/44741620)）：

+ 遍历每一个候选区域，保持浮点数边界不做量化。

+ 将候选区域分成 k×k 个区域，每个边界的单元不做量化。

+ 在每个计算单元中固定四个坐标位置，使用双线性内插的方法计算四个位置的值，然后进行最大池化操作。

  ![ROI Align](https://pic1.zhimg.com/80/v2-21ee0cf08bed9dfdca6651799e48c36e_hd.jpg)

ROI Align能使Mask精度提升10% - 50%。如果含有较多的小目标检测，使用ROI Align能获得更好的精度。

#### Loss function

多任务Loss function对于每一个ROI，有
$$
L = L_{classification} + L_{bbox} + L_{mask}
$$
其中$L_{classification}$和$L_{bbox}$与Faster R-CNN一样。mask分支每一个ROI有$k * m^2$维的输出。表示分辨率为 m * m的k个二值mask。其中k是类别数，每一类一个。对每一个像素实行一个sigmoid。定义$L_{mask}$是平均二值cross-entropy loss。对于一个ROI的ground truth是第k类。L_mask只定义在第k个mask上。（其他mask输出对于损失函数没有贡献）