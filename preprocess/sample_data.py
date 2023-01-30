import json
import numpy as np

new_obj1, new_obj2 = [], []

for is_covid in ['pos', 'neg']:
    obj1 = json.load(open(f'meta_data_info/MIDRC_cr_manifest_{is_covid}.json'))
    # obj1 = json.load(open('MIDRC_imaging_study_manifest_dx_subset.json'))
    # obj2 = json.load(open(f'meta_data_info/MIDRC_dx_manifest_{is_covid}.json'))

    l1 = len(obj1)
    rand1 = np.random.randint(0, l1, 8000)
    # rand2 = np.random.randint(0, l2, 4000)

    for idx in rand1:
        new_obj1.append(obj1[idx])

    # for idx in rand2:
    #     new_obj2.append(obj2[idx])

with open('meta_data_info/MIDRC_imaging_study_manifest_cr_subset.json', 'w') as json_out:
    json.dump(new_obj1, json_out, indent = 4)

# with open('meta_data_info/MIDRC_imaging_study_manifest_dx_subset_2.json', 'w') as json_out:
#     json.dump(new_obj2, json_out, indent = 4)