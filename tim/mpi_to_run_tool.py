
import sys
import datetime
import numpy as np
import tool_to_recog

from mpi4py import MPI

sys.path.append('/Users/mtest/anaconda3/lib/python3.7/site-packages/mpi4py')
sys.path.append('/Users/mtest/opencv_bitbucket/timestampocr')
sys.path.append('/Users/mtest/opencv_bitbucket/timestampocr')
sys.path.append('/Users/mtest/anaconda3/lib/python37.zip')
sys.path.append('/Users/mtest/anaconda3/lib/python3.7')
sys.path.append('/Users/mtest/anaconda3/lib/python3.7/lib-dynload')
sys.path.append('/Users/mtest/anaconda3/lib/python3.7/site-packages')
sys.path.append('/Users/mtest/anaconda3/lib/python3.7/site-packages/aeosa')
sys.path.append('/Users/mtest/anaconda3/lib/python3.7/site-packages/mpi4py')
sys.path.append('/Users/mtest/anaconda3/lib/python3.7/site-packages/mpi4py')


def tuple_first(elem):
    return elem[0]


print("datetime", datetime.datetime.now())
path = sys.argv[1]
platform = sys.argv[2]
#path = '/Volumes/to_save_mp4/'+path
frame_counter = tool_to_recog.get_frame_count_of_video(path)
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
print("the %d process of %d processes" % (comm_rank, comm_size))

step = int(frame_counter/(comm_size-1))
if comm_rank != comm_size-1:
    
    begin_frame = int((comm_rank)*step)
    end_frame = int((comm_rank+1)*step-1)
    # print('begin_frame'+str(comm_rank),begin_frame)
    # print('end_frame'+str(comm_rank),end_frame)
    if comm_rank == comm_size-2:
        end_frame = frame_counter-1
    dealy_ms, first_reserve_list = tool_to_recog.main_to_call(path, begin_frame, end_frame, platform,comm, comm_rank)

    comm.send([dealy_ms, first_reserve_list], dest=comm_size-1, tag=11)
    
if comm_rank == comm_size-1:
    python_path = '/Users/mtest/opencv_bitbucket/timestampocr'

    all_delay_ms = []
    all_first_reserve_list = []
    for i in range(comm_size-1):
        dealy_ms = []
        first_reserve_list = []
        [dealy_ms, first_reserve_list] = comm.recv(source=i, tag=11)
        for obj in first_reserve_list:
            all_first_reserve_list.append(obj)
        all_delay_ms.append(dealy_ms)
    # print('len(all_first_reserve_list)',len(all_first_reserve_list))
    # print('len(all_delay_ms)',len(all_delay_ms))
    dealy_ms = [0.0, 0.0, 0.0]
    i = 0
    while i < len(all_first_reserve_list):
        if i == 0 or i == len(all_first_reserve_list)-1 :
            delay = all_first_reserve_list[i][1]-all_first_reserve_list[i][0]
            pre_time = all_first_reserve_list[i][0]
            time_now = all_first_reserve_list[i][1]
            i = i+1
            
            # print('hehe1')
            # continue
        else:
            pre_image = all_first_reserve_list[i][2]
            currentImage = all_first_reserve_list[i+1][2]
            pre_time = all_first_reserve_list[i][0]
            time_now = all_first_reserve_list[i+1][1]
            # showImg('pre_image',pre_image)
            # showImg('currentImage',currentImage)
            loss3 = tool_to_recog.Loss3(platform,pre_image, currentImage, tool_to_recog.class_tool, tool_to_recog.deblur_tool)
            print('now time', all_first_reserve_list[i][0], all_first_reserve_list[i+1][1])
            if loss3 == True:
                delay = all_first_reserve_list[i+1][1]-all_first_reserve_list[i][0]
            else:
                delay = max(all_first_reserve_list[i+1][1]-all_first_reserve_list[i+1][0],
                            all_first_reserve_list[i][1]-all_first_reserve_list[i][0])
            i = i+2
        if delay >= 200:
            dealy_ms[0] = dealy_ms[0] + delay
            # print('200 pre_time is {},time_now is {},process id is {},i= {}'.format(pre_time, time_now, comm_rank, i))
        if delay >= 300:
            dealy_ms[1] = dealy_ms[1] + delay
            # print('300 pre_time is {},time_now is {},process id is {},i= {}'.format(pre_time, time_now, comm_rank, i))
        if delay >= 600:
            # print('600 pre_time is {},time_now is {},process id is {},i= {}'.format(pre_time, time_now, comm_rank, i))
            dealy_ms[2] = dealy_ms[2] + delay
        
    # print(np.sum(wode, axis = 0))
    all_delay_ms.append(dealy_ms)
    print('all_delay_ms', all_delay_ms)
    # 最终结果，数值，不是百分比
    print("final_result", np.sum(all_delay_ms, axis=0))

comm.barrier()
# print('now all thread end')
MPI.Finalize()
