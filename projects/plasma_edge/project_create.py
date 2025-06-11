# %%
import toklabel
from toklabel import ProjectBuilder
import toklabel.annotationmanage as am
import toklabel.utils

# %% 准备工作，连接Label Studio，导入配置文件，连接标注数据的数据库
plasma_shape_builder = ProjectBuilder(project_config_file='plasma_shape.yaml')

shots = plasma_shape_builder.generate_shot_list()
print(shots)
conn = am.connect_annotation_database()

anno_res = am.ts.query_annotations(conn, shots, 'ip_feature', all_info=False)

ls = toklabel.connect_Label_Studio('db485b6a000c60c389e4ba942b3827b120e3d4cb')

# %% 生成起始时间和结束时间
shot_ip_time = dict.fromkeys(shots,{'t_min':None, 't_max':None})
for anno in anno_res:
    shot_ip_time[anno['shot']]['t_min'] = anno['start_time'] - 0.01
    shot_ip_time[anno['shot']]['t_max'] = anno['end_time'] + 0.01
print(shot_ip_time)
print(shot_ip_time.values())
t_mins = [];t_maxs=[]
for shot in shots:
    t_min, t_max = shot_ip_time[shot].values()
    t_mins.append(t_min)
    t_maxs.append(t_max)
print(t_mins)
print(t_maxs)    
# %% 导出数据
plasma_shape_builder.t_max = t_maxs
plasma_shape_builder.t_min = t_mins
img_urls, img_fts = plasma_shape_builder.prepare_img()

# %% 由于项目已存在，只要同步储存就可以了

storage = plasma_shape_builder.create_storage(ls,'plasma_shapes')
print(storage)

# %%
