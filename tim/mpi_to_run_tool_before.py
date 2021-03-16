
import sys
import numpy as np
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

# print("sys.path",sys.path)
from mpi4py import MPI 
import datetime
from mpi4py import MPI 
import socket

#print(socket.gethostbyname(socket.getfqdn(socket.gethostname())))

import tool_to_recog

def tuple_first(elem):
    return elem[0]


print("datetime",datetime.datetime.now())
path = sys.argv[1]
frame_counter = tool_to_recog.get_frame_count_of_video(path)
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
print("the %d process of %d processes" %(comm_rank,comm_size))

step = int(frame_counter/(comm_size-1))
if comm_rank != comm_size-1:
    
    begin_frame = int((comm_rank)*step)
    end_frame = int((comm_rank+1)*step-1)
    # print('begin_frame'+str(comm_rank),begin_frame)
    # print('end_frame'+str(comm_rank),end_frame)
    if comm_rank == comm_size-2:
        end_frame = frame_counter-1
    dealy_ms,first_reserve_list = tool_to_recog.main_to_call(path,begin_frame,end_frame,comm,comm_rank)
    # all_first_reserve_list.append(first_reserve_list[0:-1])
    # all_delay_ms.append(dealy_ms)
    # print('len(all_delay_ms)',len(all_delay_ms))
    # print('len(all_first_reserve_list)',len(all_first_reserve_list))
    # data = zip(dealy_ms,first_reserve_list)
    comm.send([dealy_ms,first_reserve_list] ,dest = comm_size-1, tag=11)
    
if comm_rank == comm_size-1:
    python_path = '/Users/mtest/opencv_bitbucket/timestampocr'
    deblur_tool, class_tool = tool_to_recog.setUtils(
        python_path + '/esc_encoder_wokl_3.h5',
        python_path + '/ebc_encoder_wokl_3.h5',
        python_path + '/eb_encoder_wokl_3.h5',
        python_path + '/gen_s_wokl_3.h5',
        python_path + '/tenClass_cpu.h5')
    all_delay_ms = []
    all_first_reserve_list = []
    for i in range(comm_size-1):
        dealy_ms = []
        first_reserve_list = []
        [dealy_ms,first_reserve_list] = comm.recv(source=i, tag=11)
        for obj in first_reserve_list:
            all_first_reserve_list.append(obj)
        all_delay_ms.append(dealy_ms)
    # print('len(all_first_reserve_list)',len(all_first_reserve_list))
    # print('len(all_delay_ms)',len(all_delay_ms))
    dealy_ms = [0.0,0.0,0.0]
    i = 0
    while i < len(all_first_reserve_list):
        if i == 0 or i == len(all_first_reserve_list)-1 :
            delay = all_first_reserve_list[i][1]-all_first_reserve_list[i][0]
            i = i+1
            print('hehe1')
            # continue
        else :
            print('hehe')
            pre_image = all_first_reserve_list[i][2]
            currentImage = all_first_reserve_list[i+1][2]
            pre_time = all_first_reserve_list[i][0]
            time_now = all_first_reserve_list[i+1][1]
            # showImg('pre_image',pre_image)
            # showImg('currentImage',currentImage)
            loss3 = tool_to_recog.Loss3(pre_image,currentImage,class_tool,deblur_tool)#,couter_frame_wode)#,strat_list=strat_list,index_1=index_)
            print('now time',all_first_reserve_list[i][0],all_first_reserve_list[i+1][1])
            if loss3 == True:
                delay = all_first_reserve_list[i+1][1]-all_first_reserve_list[i][0]
            else :
                delay = max(all_first_reserve_list[i+1][1]-all_first_reserve_list[i+1][0], all_first_reserve_list[i][1]-all_first_reserve_list[i][0])
            i = i+2
        if delay >= 200:
            dealy_ms[0] = dealy_ms[0] + delay
            print('200 pre_time is {},time_now is {},process id is {},i= {}'.format(pre_time,time_now,comm_rank,i))
        if delay >= 300:
            dealy_ms[1] = dealy_ms[1] + delay
            print('300 pre_time is {},time_now is {},process id is {},i= {}'.format(pre_time,time_now,comm_rank,i))
        if delay >= 600:
            print('600 pre_time is {},time_now is {},process id is {},i= {}'.format(pre_time,time_now,comm_rank,i))
            dealy_ms[2] = dealy_ms[2] + delay
        
    # print(np.sum(wode, axis = 0))
    all_delay_ms.append(dealy_ms)
    print('all_delay_ms',all_delay_ms)


    print("final_result",np.sum(all_delay_ms, axis = 0))# 最终结果，数值，不是百分比

         
comm.barrier()
# print('now all thread end')
MPI.Finalize()


# if comm_rank == 0:
#     all_first_reserve_list.sort(key=tuple_first)
#     i = 0
#     while i < len(all_first_reserve_list):
#         if i == 0 or i == len(all_first_reserve_list)-1 :
#             delay = all_first_reserve_list[i][1]-all_first_reserve_list[i][0]
#         else :
#             pre_image = all_first_reserve_list[i][2]
#             currentImage = all_first_reserve_list[i+1][2]
#             # showImg('pre_image',pre_image)
#             # showImg('currentImage',currentImage)
#             loss3 = Loss3(pre_image,currentImage,class_tool,deblur_tool)#,couter_frame_wode)#,strat_list=strat_list,index_1=index_)
#             print('now time',all_first_reserve_list[i+1][1],all_first_reserve_list[i][0])
#             if loss3 == True:
#                 delay = all_first_reserve_list[i+1][1]-all_first_reserve_list[i][0]
#             else :
#                 delay = max(all_first_reserve_list[i+1][1]-all_first_reserve_list[i+1][0], first_reserve_list[i][1]-first_reserve_list[i][0])
#         if delay >= 200:
#             dealy_ms[0] = dealy_ms[0] + delay
#             # print('200 pre_time is {},time_now is {}'.format(pre_time,time_now))
#         if delay >= 300:
#             dealy_ms[1] = dealy_ms[1] + delay
#             # print('300 pre_time is {},time_now is {}'.format(pre_time,time_now))
#         if delay >= 600:
#             # print('600 pre_time is {},time_now is {}'.format(pre_time,time_now))
#             dealy_ms[2] = dealy_ms[2] + delay
#         i = i+1
#     # print(np.sum(wode, axis = 0))
#     all_delay_ms.append(dealy_ms)


# print(np.sum(all_delay_ms, axis = 0))# 最终结果，数值，不是百分比

# print(datetime.datetime.now())


# from mpi4py import MPI 
# import socket
# import sys
# print(socket.gethostbyname(socket.getfqdn(socket.gethostname())))
# sys.path.append('/Users/mtest/.local/lib/python3.7/site-packages/')
# comm = MPI.COMM_WORLD
# comm_rank = comm.Get_rank()
# comm_size = comm.Get_rank()
# print("I'm the %d process of %d processes" %(comm_rank,comm_size))
