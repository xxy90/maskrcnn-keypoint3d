import h5py
import numpy as np
import json
## h5file to COCO jason

def ToJason():
    h36mImgDir = '/mnt/share/human36/pre_process/images'
    f=h5py.File('d:/360/mask/annotSampleTest.h5','r')
    w_path = 'd:/360/mask/train.json'


    print(f.keys())#['action', 'bbox', 'camera', 'id', 'istrain', 'joint_2d', 'joint_3d_mono', 'subaction', 'subject']
    keys=[ key for key in f.keys()]
    print(keys)

    ## get trainble data
    label = f['istrain'][:].tolist()
    label_index = [i for i, x in enumerate(label) if x == 1]
    category_id = 1

    is_crowd=0

    with open(w_path, 'w+') as ff:
        for index in label_index:
            folder = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(f['subject'][index], f['action'][index],
                                                                          f['subaction'][index], f['camera'][index])
            img_name = '{}/{}/{}_{:06d}.jpg'.format(h36mImgDir, folder, folder, f['id'][index])
            keypoints = f['joint_2d'][index]
            num_keypoints = len(keypoints)


            print('the num_keypoints is ',num_keypoints)
            keypoints_2d = insert_v_point(keypoints)
            keypoints_3d = f['joint_3d_mono'][index]
            keypoints3d=round_list(keypoints_3d)
            #print('the type of keypoints_3d is ',type(keypoints_3d),type(keypoints_2d))
            bbox = extract_box(keypoints)

            djason = {"category_id": category_id, "num_keypoints": num_keypoints, "is_crowd": is_crowd,
                      "keypoints": keypoints_2d,
                      "keypoints3D": keypoints3d, "bbox": bbox,
                      "image_id": img_name
                      }
            jstr = json.dumps(djason, ensure_ascii=False)






            ff.write(jstr)
    ff.close()

    return None


def ToJason_val():
    h36mImgDir = '/mnt/share/human36/pre_process/images'
    f=h5py.File('d:/360/mask/annotSampleTest.h5','r')
    w_path = 'd:/360/mask/val.json'


    print(f.keys())#['action', 'bbox', 'camera', 'id', 'istrain', 'joint_2d', 'joint_3d_mono', 'subaction', 'subject']
    keys=[ key for key in f.keys()]
    print(keys)

    ## get trainble data
    label = f['istrain'][:].tolist()
    label_index = [i for i, x in enumerate(label) if x == 0]
    category_id = 1

    is_crowd=0

    with open(w_path, 'w+') as ff:
        for index in label_index:
            folder = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(f['subject'][index], f['action'][index],
                                                                          f['subaction'][index], f['camera'][index])
            img_name = '{}/{}/{}_{:06d}.jpg'.format(h36mImgDir, folder, folder, f['id'][index])
            keypoints = f['joint_2d'][index]
            num_keypoints = len(keypoints)


            print('the num_keypoints is ',num_keypoints)
            keypoints_2d = insert_v_point(keypoints)
            keypoints_3d = f['joint_3d_mono'][index]
            keypoints3d=round_list(keypoints_3d)
            #print('the type of keypoints_3d is ',type(keypoints_3d),type(keypoints_2d))
            bbox = extract_box(keypoints)

            djason = {"category_id": category_id, "num_keypoints": num_keypoints, "is_crowd": is_crowd,
                      "keypoints": keypoints_2d,
                      "keypoints3D": keypoints3d, "bbox": bbox,
                      "image_id": img_name
                      }
            jstr = json.dumps(djason, ensure_ascii=False)






            ff.write(jstr)
    ff.close()

    return None
def round_list(points):
    ## round the points to 2 decimal and transform to list
    r=[]
    for i in range(len(points)):
        r.append(round(points[i][0],2))
        r.append(round(points[i][1], 2))
        r.append(round(points[i][2], 2))

    return r


def insert_v_point(points):
    #append the visbility to keypoint i.e [30,30] to[30,30,v],
    # v=[0,1,2],0 is not labeled and not visible ,1 is labeled and not visible ,2 is visible
    v=[]

    for i in range(len(points)):

        if points[i].all()==0:
            v.append(round(points[i][0],2))
            v.append(round(points[i][1], 2))

            v.extend(0)
        else:
            v.append(round(points[i][0], 2))
            v.append(round(points[i][1], 2))
            v.append(2)

    #print(v)

    return v




def extract_box(points):
    ## get the bbox from keypoints

    min_x = int(min([points[ i][0] for i in range(16)]))
    max_x = int(max([points[ i][0] for i in range(16)]))
    min_y = int(min([points[ i ][1] for i in range(16)]))
    max_y = int(max([points[ i ][1] for i in range(16)]))

    return [min_x, min_y, max_x, max_y]



if __name__=='__main__':
    ToJason()
    ToJason_val()










