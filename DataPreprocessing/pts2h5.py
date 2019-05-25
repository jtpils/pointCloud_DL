import numpy as np
import random
import pickle
from hdf5_util import *


#输出num*3
def random_sample_idx(points, pn):
    output_points = np.zeros((pn, 3), dtype=np.float32)
    input_size = points.shape[0]
    idx = np.zeros(input_size)
    for rand_i in range(pn):
        while 1:
            rand_pos=random.randint(0, input_size) % input_size
            if idx[rand_pos]==0:
                idx[rand_pos]=1
                output_points[rand_i][0]=points[rand_pos][0]
                output_points[rand_i][1]=points[rand_pos][1]
                output_points[rand_i][2]=points[rand_pos][2]
                break
    return output_points


#数据采样至规定数据量
def pc_augment_to_point_num(x,y,z, pn):
    point=np.stack((x, y, z), axis=1)
    cur_len = x.__len__()
    if x.__len__() <= pn:
        res = np.array(point)
        while cur_len < pn:
            res = np.concatenate((res, point))
            cur_len += x.__len__()
        return res[:pn, :]
    else:
        #降采样
        point=random_sample_idx(point, pn)
        return point


def point_cloud2batch(input_file):
    point_num = 4096
    point_cloud_str = [line.rstrip() for line in open(input_file, 'r')]
    point_cloud = np.array([np.float32(s.split()) for s in point_cloud_str], dtype=np.float32)
    #只取XYZ三维坐标
    point_cloud = point_cloud[:, 0:3]
    max = point_cloud.max(axis=0)
    min = point_cloud.min(axis=0)
    delta = max - min + 1
    delta_arr = np.array(delta)
    delta_arr = delta_arr.astype(np.int32)
    print("delta_XYZ=", delta)
    print("point cloud shape is:", point_cloud.shape)
    #分成1m*1m的batch数据
    grid_size= delta_arr[0]*delta_arr[1]*delta_arr[2]
    #flag=np.zeros(delta_arr)
    list_x = []
    list_y = []
    list_z = []
    print("gridsize=",grid_size)
    for i_init in range(grid_size):
        list_x.append([])
        list_y.append([])
        list_z.append([])
    for i in range(point_cloud.shape[0]):
        x_pos = np.int32((point_cloud[i,0]-min[0])//1)
        y_pos = np.int32((point_cloud[i,1]-min[1])//1)
        z_pos = np.int32((point_cloud[i,2]-min[2])//1)
        pos = x_pos+y_pos*delta_arr[0]+z_pos*delta_arr[0]*delta_arr[1]
        #print(pos)
        #flag[pos]+=1
        list_x[pos].append(point_cloud[i,0])
        list_y[pos].append(point_cloud[i,1])
        list_z[pos].append(point_cloud[i,2])
    #resample list B*N*3
    out_data = []
    for resample_i in range(grid_size):
        if list_x[resample_i].__len__() == 0:
            continue
        else:
            one_batch = pc_augment_to_point_num(list_x[resample_i], list_y[resample_i], list_z[resample_i], point_num)
            out_data.append(one_batch)
    out_data = np.array(out_data)
    print("batch data shape is:", out_data.shape)
    return out_data

#with_RGB_pos = 1 or 0
def write_tensor_label_hdf5(input_file,with_RGB_pos):
    """ We will use constant label (class 0) for the test data """
    # set batch buffer
    if with_RGB_pos==1:
        h5_batch_data = point_cloud2batch_add_RGB_position(input_file)
    else:
        h5_batch_data = point_cloud2batch(input_file)

    (file_path, temp_filename) = os.path.split(input_file);
    filename = os.path.splitext(temp_filename)[0]
    if with_RGB_pos==1:
        filename = filename + '_with_RGB_pos'
    output_filename_prefix = 'volume_data'
    h5_filename = output_filename_prefix + '_' + filename + '.h5'
    data_type = "float32"
    print(h5_filename)
    print(np.shape(h5_batch_data))
    print(h5_filename, data_type)
    #建立label
    #必须指定start和end batch位置
    end=h5_batch_data.shape[0]
    save_h5_data(h5_filename, h5_batch_data[0:end,:,:], data_type)
    return


def write_tensor_label_pickle(input_file,with_RGB_pos):
    """ We will use constant label (class 0) for the test data """
    # set batch buffer
    if with_RGB_pos == 1:
        pickle_batch_data = point_cloud2batch_add_RGB_position(input_file)
    else:
        pickle_batch_data = point_cloud2batch(input_file)
    (file_path, temp_filename) = os.path.split(input_file);
    filename = os.path.splitext(temp_filename)[0]
    if with_RGB_pos==1:
        filename = filename + '_with_RGB_pos'
    output_filename_prefix = 'volume_data'
    pickle_filename = output_filename_prefix + '_' + filename + '.pickle'
    data_type = "float32"
    print(pickle_filename)
    print(np.shape(pickle_batch_data))
    print(pickle_filename, data_type)
    #建立label
    #必须指定start和end batch位置
    #end=pickle_batch_data.shape[0]
    with open(pickle_filename, 'wb') as fp:
        pickle.dump(pickle_batch_data, fp, protocol=2)
    return

#输出num*9
def random_sample_idx_add_RGB_position(points, pn):
    output_points = np.zeros((pn, 9), dtype=np.float32)
    input_size = points.shape[0]
    idx = np.zeros(input_size)
    for rand_i in range(pn):
        while 1:
            rand_pos=random.randint(0, input_size) % input_size
            if idx[rand_pos]==0:
                idx[rand_pos]=1
                output_points[rand_i][0]=points[rand_pos][0]
                output_points[rand_i][1]=points[rand_pos][1]
                output_points[rand_i][2]=points[rand_pos][2]
                output_points[rand_i][3]=points[rand_pos][3]
                output_points[rand_i][4]=points[rand_pos][4]
                output_points[rand_i][5]=points[rand_pos][5]
                output_points[rand_i][6]=points[rand_pos][6]
                output_points[rand_i][7]=points[rand_pos][7]
                output_points[rand_i][8]=points[rand_pos][8]
                break
    return output_points

#数据采样至规定数据量
def pc_augment_to_point_num_add_RGB_position(x,y,z,R,G,B,Rx,Ry,Rz,pn):
    point=np.stack((x, y, z,R,G,B,Rx,Ry,Rz), axis=1)
    cur_len = x.__len__()
    if x.__len__() <= pn:
        res = np.array(point)
        while cur_len < pn:
            res = np.concatenate((res, point))
            cur_len += x.__len__()
        return res[:pn, :]
    else:
        #降采样
        point=random_sample_idx_add_RGB_position(point, pn)
        return point

#XYZ+RGB+relative positon
def point_cloud2batch_add_RGB_position(input_file):
    point_num = 4096
    point_cloud_str = [line.rstrip() for line in open(input_file, 'r')]
    point_cloud = np.array([np.float32(s.split()) for s in point_cloud_str], dtype=np.float32)
    #只取XYZ三维坐标
    point_cloud = point_cloud[:, 0:3]
    row=np.shape(point_cloud)
    row=row[0]
    max = point_cloud.max(axis=0)
    min = point_cloud.min(axis=0)
    delta = max - min + 1
    delta_arr = np.array(delta)
    delta_arr = delta_arr.astype(np.int32)

    lwh=max-min
    relative_x=(point_cloud[:,0]-min[0])/lwh[0]
    relative_y = (point_cloud[:, 1] - min[1]) / lwh[1]
    relative_z = (point_cloud[:, 2] - min[2]) / lwh[2]
    relative_x=relative_x.reshape(-1,1)
    relative_y=relative_y.reshape(-1,1)
    relative_z=relative_z.reshape(-1,1)
    relative_pos=np.concatenate((relative_x,relative_y,relative_z),axis=1)
    RGB=np.zeros((row,3))
    point_cloud=np.concatenate((point_cloud,RGB,relative_pos),axis=1)
    print("delta_XYZ=", delta)
    print("point cloud shape is:", point_cloud.shape)
    #分成1m*1m的batch数据
    grid_size= delta_arr[0]*delta_arr[1]*delta_arr[2]
    #flag=np.zeros(delta_arr)
    list_x = []
    list_y = []
    list_z = []
    list_R = []
    list_G = []
    list_B = []
    list_Rx = []
    list_Ry = []
    list_Rz = []
    print("gridsize=",grid_size)
    for i_init in range(grid_size):
        list_x.append([])
        list_y.append([])
        list_z.append([])
        list_R.append([])
        list_G.append([])
        list_B.append([])
        list_Rx.append([])
        list_Ry.append([])
        list_Rz.append([])
    for i in range(point_cloud.shape[0]):
        x_pos = np.int32((point_cloud[i,0]-min[0])//1)
        y_pos = np.int32((point_cloud[i,1]-min[1])//1)
        z_pos = np.int32((point_cloud[i,2]-min[2])//1)
        pos = x_pos+y_pos*delta_arr[0]+z_pos*delta_arr[0]*delta_arr[1]

        list_x[pos].append(point_cloud[i,0])
        list_y[pos].append(point_cloud[i,1])
        list_z[pos].append(point_cloud[i,2])
        list_R[pos].append(point_cloud[i,3])
        list_G[pos].append(point_cloud[i,4])
        list_B[pos].append(point_cloud[i,5])
        list_Rx[pos].append(point_cloud[i,6])
        list_Ry[pos].append(point_cloud[i,7])
        list_Rz[pos].append(point_cloud[i,8])
    #resample list B*N*3
    out_data = []
    for resample_i in range(grid_size):
        if list_x[resample_i].__len__() == 0:
            continue
        else:
            one_batch = pc_augment_to_point_num_add_RGB_position(list_x[resample_i], list_y[resample_i], list_z[resample_i], \
                                                list_R[resample_i],list_G[resample_i],list_B[resample_i], \
                                                list_Rx[resample_i],list_Ry[resample_i],list_Rz[resample_i],point_num)
            out_data.append(one_batch)
    out_data = np.array(out_data)
    print("batch data shape is:", out_data.shape)
    return out_data

def main():
    input_file = "D:/data/8212after_seg/3.txt"
    #input_file = "D:/part_pointcloud.txt"
    #write_tensor_label_hdf5(input_file,0)
    write_tensor_label_pickle(input_file,0)

if __name__ == '__main__':
    main()
    '''
    print("reading h5 start:")
    
    filename="volume_data_3.h5"
    d = load_h5_data(filename)
    print(d.shape)
    '''
    filename="volume_data_3.pickle"
    print("reading pickle start:")
    with open(filename, 'rb') as fp:
        data3 = pickle.load(fp)
    # 此处使用的是load(目标文件)
    #data3 = pickle.load(fp)
    print(data3.shape)
    # print(__name__)
