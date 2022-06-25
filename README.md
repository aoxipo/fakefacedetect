# fakefacedetect
天池大数据比赛 伪造人脸攻击图像区分检测
# model
考虑到人脸在数据集分布在原始图像上偏离中间50%-60%范围对超像素图片切分为3*3，再使用dense121做特征提取，最后合并结果对结果使用自注意力进行全连接层分类
# dataset
为了降低模型对高分辨率的依赖，我们对数据进行线性插值2次resize，以均匀分布概率分别resize到32 64 128 256 大小再还原到原始尺寸。
# 结果
在测试集上获取了97%的准确率
在天池排行榜上排名60名
