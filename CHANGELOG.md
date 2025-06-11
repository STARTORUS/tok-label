# Change Log

## 2025-04-30

change:

- 独立ip_features模型服务，迁移到<https://gitlab.startorus.org/scientific-computing/ml-models>
- frame_extractor导出图片数据时，会同时导出帧时间，帧时间也将被上传到Label Studio的图像数据标注项目中，便于后续数据处理和数据筛选
- 对于一组shot的图片提取，支持为每一炮设置不同的起始时间和结束时间。这一点通过允许t_min和t_max为list实现

## 2025-05-01

change:

- 重构toklabel.create_label_config, 将一些仅仅为创建label_config服务的辅助函数移到了_label_config_utils.py中。支持多种类的标签的基础配置以及使用SAM2的特殊UI风格。 该功能的定位是完成基础配置，减少繁琐的构造XML格式字符串的过程。用户可以在Label Studio的UI界面添加更多自定义属性。

## 2025-05-02

change:

- 重构toklabel.image_json_convertor,将解析具体标注内容的部分独立出来，便于接入新的标注格式。该改动针对图像标注种类多样的问题，目前已经支持解析矩形框、关键点和刷标签(mask)的标注结果。
- 在Prediction中增加了新的数据类DataContext，用于存储frame_time, shot等信息。在某些情况下，这些参数可能会发挥作用。据此，修改了BasePredictor，toklabel.import_prediction(已重命名为import_prediction_timeseries);并增加了import_prediction_image。现在，导入预标注的流程更加完善了。

## 2025-05-03

change:

- 调整的annotationmanage的结构，将其作为toklabel的一个子包，将时序标注数据和图像标注数据的相关接口分别放在两个文件下。
- 初步完成了图像标注数据的存储和管理函数，尚未使用数据测试。

## 2025-05-04

change:

- 加入了使用polygonlabel(多边形)的标注任务，toklabel也支持处理polygonlabels；加入了生成等离子体中心，并支持以keypoint的形式输出到label studio。

## 2025-05-07

change:

- 基本完成对annotationmanage的结构的调整，实现了对图像标注数据的提取、存储以及查找，保持了较高的可拓展性。目前的设计是将数字标注和其他图像标注分开存储，前者可以线性插值得到更精细的结果，而后者只能保持原始数据。将原来属于时序部分的部分通用函数提出与图像部分的通用函数一起，置于annotationmanage.utils.py，使得代码结构更清晰。
