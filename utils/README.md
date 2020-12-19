- 支持的损失函数
    - CE_Loss: 用于多分类（包括二分类）的交叉熵损失函数
    - W_CELoss: 带权重的交叉熵损失，可以平衡原本数量
    - FocalLoss: 用于多分类（包括二分类）的Focal Loss
    - OHEM_CE_Loss: OHEN交叉熵损失函数

- 支持的评价指标
    - Acc
    - mAP
    - f1score
    
- 支持的学习率配置
    - exponential: 指数衰减
    - cosine_decay: 余弦衰减
    - cosine_decay_restart: 重启余弦衰减
    - linear_cosine_decay: 线性余弦衰减
    - noise_linear_cosine_decay: 噪声线性余弦衰减
    - inverse_time_decay: 倒数衰减
    - fixed: 固定学习率
    - piecewise: 分段常数衰减
    
- 支持的优化器:
    - Adam: Adam算法，自适应学习率
    - sgd: 梯度下降法
    - RMSProp: RMSProp算法，自适应学习率
    - Momentum: 动量优化法,一般动量momentum取0.9
    
- 支持的数据增广
    - 基础数据增广,颜色方面（调节对比度，调节亮度，调节Hue，添加饱和度，添加高斯噪声等
    - 随机旋转
    - 随机缩放
    - 翻转
    - CutMix
    - Cutout
    - GridMask
    - MixUp
    - RandomErasing: 随机擦除
    - random_zoom: 随机缩放, 居中裁剪
    
    
    
    
- util.py
    - add_regularization: 添加正则损失
    