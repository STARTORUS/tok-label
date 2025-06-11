# %%
import toklabel.utils as utils
import toklabel.maskutils as mask_utils
import numpy as np
import toklabel
import cv2
from toklabel.annotationmanage import connect_annotation_database, img
# %% 导出并解析标注数据
ls = toklabel.connect_Label_Studio('db485b6a000c60c389e4ba942b3827b120e3d4cb')
anno = toklabel.export_annotation(ls,project_id=47, json_min=False, exclude_skipped=False)
anno_parsed  = toklabel.image_json_convertor(anno,json_min=False, taking_skipped_as_label=True, multi_label_group=True)
anno_geom = []
anno_num = []
rename_map = {
    'label': 'position',
    'feature': 'label',
}
for a in anno_parsed:        
    if a.get('geom_type'):
        if a.get('geom_type') == 'number':
            number_data  = a['geom_data']
            number_anno = a.copy()
            del number_anno['geom_type']
            del number_anno['geom_data']
            number_anno.update(number_data)
            print(number_anno)
            anno_num.append(number_anno)
        else:
            for old_key, new_key in rename_map.items():
                a[new_key] = a.pop(old_key)
            a.pop('image_width')
            a.pop('image_height')
            a.pop('shot')
            anno_geom.append(a)   
    else:
        a.pop('shot')
        anno_geom.append(a)
# %% 建表，数字和图像标注分开存储
pg_conn = connect_annotation_database()
shot = 240830026
img.create_image_table(pg_conn, shot, 'shape', ['label','position'])
img.create_number_table(pg_conn, shot, 'parameters', ['R','Z','a','k','delta_u','delta_l'],[])
# %% 插入数据
img.insert_annotations(pg_conn, shot, 'shape', anno_geom,
                       on_conflict=('frame_id','position','geom_type','annotator'))
img.insert_annotations(pg_conn, shot,'parameters',anno_num,
                       on_conflict=('frame_id','position','geom_type','annotator'))

# %% 查询标注数据例子一
res = img.query_image_annotations(pg_conn, shot, 'shape', 
                                  start_time=0.5, end_time=0.65, 
                                  filters={"position":"left"},
                                  columns=["frame_time", "geom_type", "geom_data","label", "annotator", "annotation_id"])
for result in res:
    print(result)

# %% 例子二
# number支持线性插值，这样较粗糙的标注也能获得较高的时间分辨率
t,data,columns = img.query_number_table(pg_conn, shot, 'parameters',
                                        as_array=True, interpolate=True, target_resolution=1e-4)
print(columns)
print(data.shape)
print(t.shape,t[0],t[-1])