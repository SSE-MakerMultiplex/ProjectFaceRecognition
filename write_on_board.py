import sys
import os
import time
import threading
import argparse
import csv
import numpy as np

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check
    
def example_angular_action_movement_1(base,angles):
    
    print("Starting angular action movement ...")
    action = Base_pb2.Action()
    action.name = "Example angular action movement"
    action.application_data = ""

    actuator_count = base.GetActuatorCount()

    # Place arm straight up
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = angles[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def example_cartesian_action_movement(base, position,orientation,velocity):
    
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    #feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    #speed
    speed=action.reach_pose.constraint.speed
    speed.translation=velocity
    #
    cartesian_pose.x = position[0]         # (meters)
    cartesian_pose.y = position[1]    # (meters)
    cartesian_pose.z = position[2]    # (meters)
    cartesian_pose.theta_x = orientation[0] # (degrees)
    cartesian_pose.theta_y = orientation[1] # (degrees)
    cartesian_pose.theta_z = orientation[2] # (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def read_csv(filename,angle,position,orientation,gripper_position,translation_speed,action_sequence):
    first_row_checker=0
    single_row=[]
    with open(filename,newline='') as csvfile:
        reader=csv.reader(csvfile,delimiter=',')
        for row in reader:
            if first_row_checker==0:
                first_row_checker=first_row_checker+1
                continue
            if row[4]=='7':
                action_sequence.append(int(row[4]))
                single_row.append(float(row[5]))
                single_row.append(float(row[7]))
                single_row.append(float(row[9]))
                single_row.append(float(row[11]))
                single_row.append(float(row[13]))
                single_row.append(float(row[15]))
                single_row.append(float(row[17]))
                angle.append(single_row)
                single_row=[]
            elif row[4]=='6':
                print(position)
                action_sequence.append(int(row[4]))
                single_row.append(float(row[23]))
                single_row.append(float(row[24]))
                single_row.append(float(row[25]))
                position.append(single_row)
                single_row=[]
                print(orientation)
                single_row.append(float(row[26]))
                single_row.append(float(row[27]))
                single_row.append(float(row[28]))
                orientation.append(single_row)
                single_row=[]
                translation_speed.append(float(row[29]))
            else:
                action_sequence.append(int(row[4]))
                gripper_position.append(float(row[32]))

def main():
    
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities
    

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        #variable
        marker_pickup_angle=[]
        marker_pickup_position=[]
        marker_pickup_orientation=[]
        marker_pickup_gripper_position=[]
        marker_pickup_translation_speed=[]
        marker_pickup_action_sequence=[]

        angle=[]
        position=[]
        orientation=[]
        gripper_position=[]
        translation_speed=[]
        action_sequence=[]

        marker_placement_angle=[]
        marker_placement_position=[]
        marker_placement_orientation=[]
        marker_placement_gripper_position=[]
        marker_placement_translation_speed=[]
        marker_placement_action_sequence=[]

        new_angle = new_position = new_orientation = gripper_sequence = speedcounter = 0
        Filename_Pickup= '/home/dospina/Kinova_CV/FaceRecognition_NewStandPU.csv'
        Filename_Placing='/home/dospina/Kinova_CV/FaceRecognition_NewStandDO.csv'
        # Create required services
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        #gripper services
         # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        
        #reading the marker picking csv file
        read_csv(Filename_Pickup,marker_pickup_angle,marker_pickup_position,marker_pickup_orientation,marker_pickup_gripper_position,marker_pickup_translation_speed,marker_pickup_action_sequence)

        # Example core
        success = True
        #performing the picking action
        for i in range(len(marker_pickup_action_sequence)):
            if marker_pickup_action_sequence[i]==7:
                success &= example_angular_action_movement_1(base,marker_pickup_angle[new_angle])
                new_angle=new_angle+1
            elif marker_pickup_action_sequence[i]==6:
                success &= example_cartesian_action_movement(base,marker_pickup_position[new_position],marker_pickup_orientation[new_orientation],marker_pickup_translation_speed[speedcounter])
                new_orientation=new_orientation+1
                new_position=new_position+1
                speedcounter=speedcounter+1
            else:
                finger.value=marker_pickup_gripper_position[gripper_sequence]
                base.SendGripperCommand(gripper_command)
                time.sleep(1)
                gripper_sequence=gripper_sequence+1
        #setting all counter to zero for reusing
        opposite_home=[182.395,15.008,180,230,0,54.997,89.999]
        

        start_position1=np.asarray(args.start_position)
        start_position1 = start_position1.astype(np.float)
        start_position1=start_position1.tolist()
        
        if start_position1[0]<0:
            success &= example_angular_action_movement_1(base,opposite_home)
        new_angle=new_position=new_orientation=gripper_sequence=speedcounter=0
        #changes from here
        
        ori=np.asarray(args.orientation)
        ori = ori.astype(np.float)
        ori=ori.tolist()

        A=[
            [0,0,0],
            [0.1,-0.02,0],
            [0,-0.04,0],
            [0.03,-0.01,0.05],  
            [0.03,-0.01,0],
            [0.03,-0.03,0],
            [0,-0.04,0.05]
        ]
        B=[
            [0,0,0],
            [0,-0.04,0],#3 start here
            [0.01,-0.05,0],
            [0.04,-0.05,0],
            [0.05,-0.04,0],
            [0.05,0,0],
            [0.05,0,0.05],
            [0.05,-0.04,0.05],
            [0.05,-0.04,0],
            [0.06,-0.05,0],
            [0.09,-0.05,0],
            [0.1,-0.04,0],
            [0.1,0,0],
            [0,0,0],
            [0,0,0.05],
            [0,-0.05,0.05]
        ]

        C=[
            [0,-0.04,0],
            [0,-0.015,0],
            [0.01,-0.005,0],
            [0.09,-0.005,0],
            [0.1,-0.015,0],
            [0.1,-0.04,0],
            [0,-0.05,0.05]
        ]

        D=[
            [0,0,0],
            [0.1,0,0],
            [0.1,-0.02,0],
            [0.08,-0.04,0],
            [0.02,-0.04,0],
            [0,-0.02,0],
            [0,0,0],
            [0,-0.04,0.05]
        ]

        E=[
            [0,0,0],
            [0.1,0,0],
            [0.1,-0.04,0],
            [0.1,-0.04,0.05],
            [0.05,0,0],
            [0.05,-0.03,0],
            [0.05,-0.03,0.05],
            [0,0,0],
            [0,-0.04,0],
            [0,-0.04,0.05]
        ]

        F=[
            [0,0,0],
            [0.1,0,0],
            [0.1,-0.04,0],
            [0.1,-0.04,0.05],
            [0.05,0,0],
            [0.05,-0.03,0],
            [0.05,-0.03,0.05],
            [0,-0.04,0.05]
        ]
        #need to check
        G=[
            [0.07,-0.05,0],
            [0.09,-0.05,0],#code for G start here
            [0.1,-0.045,0],
            [0.1,-0.015,0],
            [0.09,-0.01,0],
            [0.07,0,0],
            [0.03,0,0],
            [0.01,-0.01,0],
            [0,-0.015,0],
            [0,-0.035,0],
            [0.01,-0.04,0],
            [0.03,-0.04,0],
            [0.03,-0.02,0],
            [0.03,-0.02,0.05],
            [0,-0.04,0.05]
        ]

        H=[
            [0,0,0],
            [0.1,0,0],
            [0.1,0,0.05],
            [0.05,0,0.05],
            [0.05,0,0],
            [0.05,-0.04,0],
            [0.05,-0.04,0.05],
            [0.1,-0.04,0],
            [0,-0.04,0],
            [0,-0.04,0.05]
        ]

        I=[
            [0,0,0],
            [0,-0.04,0],
            [0,-0.04,0.05],
            [0,-0.02,0],
            [0.1,-0.02,0],
            [0.1,-0.02,0.05],
            [0.1,0,0.05],
            [0.1,0,0],
            [0.1,-0.04,0],
            [0.1,-0.04,0.05],
            [0,-0.04,0.05]
        ]

        J=[
            [0.1,0,0.05],
            [0.1,0,0],
            [0.1,-0.04,0],
            [0.1,-0.04,0.05],
            [0.1,-0.03,0],
            [0.01,-0.03,0],
            [0,0.02,0],
            [0,0.01,0],
            [0.01,0,0],
            [0.01,0,0.05],
            [0,0.05,0.05]
        ]

        K=[
            [0,0,0],
            [0.1,0,0],
            [0.1,0,0.05],
            [0.04,0,0.05],
            [0.04,0,0],
            [0.1,-0.04,0],
            [0.1,-0.04,0.05],
            [0.04,0,0],
            [0,-0.04,0],
            [0,-0.04,0.05]
        ]

        L=[
            [0,0,0],
            [0.1,0,0],
            [0.1,0,0.05],
            [0,0,0],
            [0,-0.04,0],
            [0,-0.04,0.05], 
        ]

        M=[
            [0,0,0],
            [0.1,0,0],
            [0.06,-0.02,0],
            [0.1,-0.04,0],
            [0,-0.04,0],
            [0,-0.04,0.05]
        ]

        N=[
            [0,0,0],
            [0.1,0,0],
            [0,-0.04,0],
            [0.1,-0.04,0],
            [0,-0.04,0.05]
        ]
        #marker slide
        O=[
            [0,-0.02,0],
            [0,-0.05,0],
            [0.02,-0.07,0],
            [0.08,-0.07,0],
            [0.1,-0.05,0],
            [0.1,-0.02,0],
            [0.08,0,0],
            [0.02,0,0],
            [0,-0.02,0],
            [0,-0.02,0.05],
            [0,-0.065,0.05]
        ]

        P=[
            [0,0,0],
            [0.1,0,0],
            [0.1,-0.03,0],
            [0.09,-0.04,0],
            [0.06,-0.04,0],
            [0.05,-0.03,0],
            [0.05,0,0],
            [0.05,0,0.05],
            [0,-0.05,0.05]
        ]

        # need to check manually
        Q=[
            [0,-0.02,0],
            [0,-0.05,0],
            [0.02,-0.07,0],
            [0.08,-0.07,0],
            [0.1,-0.05,0],
            [0.1,-0.02,0],
            [0.08,0,0],
            [0.02,0,0],
            [0,-0.02,0],
            [0,-0.02,0.05],
            [0.01,-0.04,0.05],
            [0.01,-0.04,0],
            [0,-0.06,0],
            [0,-0.06,0.05],
            [0,-0.08,0.05]
        ]

        R=[
            [0,0,0],
            [0.1,0,0],
            [0.1,-0.03,0],
            [0.09,-0.04,0],
            [0.06,-0.04,0],
            [0.05,-0.03,0],
            [0.05,0,0],
            [0,-0.04,0],
            [0,-0.04,0.05]
        ]

        S=[
            [0,0,0],
            [0,-0.03,0],
            [0.01,-0.04,0],
            [0.04,-0.04,0],
            [0.05,-0.03,0],
            [0.05,-0.01,0],#start here
            [0.06,0,0],
            [0.09,0,0],
            [0.1,-0.01,0],
            [0.1,-0.04,0],
            [0.1,-0.04,0.05],
            [0,-0.04,0.05]
        ]

        T=[
            [0,-0.02,0.05],
            [0,-0.02,0],
            [0.1,-0.02,0],
            [0.1,-0.02,0.05],
            [0.1,0,0.05],
            [0.1,0,0],
            [0.1,-0.04,0],
            [0.1,-0.04,0.05],
            [0,-0.04,00.05]
        ]

        U=[
            [0,-0.01,0],
            [0,-0.03,0],
            [0.01,-0.04,0],
            [0.1,-0.04,0],
            [0.1,-0.04,0.05],
            [0.1,0,0.05],
            [0.1,0,0],
            [0.01,0,0],
            [0,-0.01,0],
            [0,-0.01,0.05],
            [0,-0.05,0.05]
        ]

        V=[
            [0,-0.02,0],
            [0.1,0,0],
            [0.1,0,0.05],
            [0.1,-0.04,0.05],
            [0.1,-0.04,0],
            [0,-0.02,0],
            [0,-0.02,0.05],
            [0,-0.05,0.05]
        ]

        W=[
            [0.1,0,0],
            [0,-0.01,0],
            [0.02,-0.02,0],
            [0,-0.04,0],
            [0.1,-0.05,0],
            [0.1,-0.05,0.05],
            [0,-0.05,0.05]
        ]

        X=[
            [0,0,0],
            [0.1,-0.04,0],
            [0.1,-0.04,0.05],
            [0.1,0,0.05],
            [0.1,0,0],
            [0,-0.04,0],
            [0,-0.04,0.05]
        ]

        Y=[
            [0,-0.02,0],
            [0.05,-0.02,0],
            [0.1,0,0],
            [0.1,0,0.05],
            [0.1,-0.04,0.05],
            [0.1,-0.04,0],
            [0.05,-0.02,0],
            [0.05,-0.02,0.05],
            [0,-0.05,0.05]
        ]

        Z=[
            [0.1,0,0],
            [0.1,-0.04,0],
            [0,0,0],
            [0,-0.04,0],
            [0,-0.4,0.05]
        ]
        

        velocity1=0.1
        #matix
        last_row=[0,0,0,1]
        np_vect1=np.asarray(args.vect1)
        np_vect1 = np_vect1.astype(np.float)
        np_vect2=np.asarray(args.vect2)
        np_vect2 = np_vect2.astype(np.float)
        np_vect3=np.asarray(args.vect3)
        np_vect3 = np_vect3.astype(np.float)
        np_start_position=np.asarray(args.start_position)
        np_start_position = np_start_position.astype(np.float)
        matrix=np.vstack((np_vect1,np_vect2,np_vect3,np_start_position))
        matrix=np.transpose(matrix)
        matrix=np.vstack((matrix,last_row))
        np_normal_vector=np.asarray(args.normal_vector)
        normal_vector = np_normal_vector.astype(np.float)
        print("I am here")
        
        for i,p in zip(args.name, range(len(args.name))):
            if i=="A":
            ####think about z
                np_A=np.asarray(A)
                np_A_trans=np.transpose(np_A)
                one=np.ones((np_A.shape[0],), dtype=int)
                np_A_trans=np.vstack((np_A_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                A_mul=np.matmul(matrix,np_A_trans)
                A_mul_list=((np.transpose(A_mul))[:,0:3]).tolist()
                for j in range(len(A_mul_list)):
                    success &= example_cartesian_action_movement(base,A_mul_list[j],ori,velocity1)
            if i=="B":
                np_B=np.asarray(B)
                np_B_trans=np.transpose(np_B)
                one=np.ones((np_B.shape[0],), dtype=int)
                np_B_trans=np.vstack((np_B_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                B_mul=np.matmul(matrix,np_B_trans)
                B_mul_list=((np.transpose(B_mul))[:,0:3]).tolist()
                for j in range(len(B_mul_list)):
                    success &= example_cartesian_action_movement(base,B_mul_list[j],ori,velocity1)
            if i=="C":
                np_C=np.asarray(C)
                np_C_trans=np.transpose(np_C)
                one=np.ones((np_C.shape[0],), dtype=int)
                np_C_trans=np.vstack((np_C_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                C_mul=np.matmul(matrix,np_C_trans)
                C_mul_list=((np.transpose(C_mul))[:,0:3]).tolist()
                for j in range(len(C_mul_list)):
                    success &= example_cartesian_action_movement(base,C_mul_list[j],ori,velocity1)
            if i=="D":
                np_D=np.asarray(D)
                np_D_trans=np.transpose(np_D)
                one=np.ones((np_D.shape[0],), dtype=int)
                np_D_trans=np.vstack((np_D_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                D_mul=np.matmul(matrix,np_D_trans)
                D_mul_list=((np.transpose(D_mul))[:,0:3]).tolist()
                for j in range(len(D_mul_list)):
                    success &= example_cartesian_action_movement(base,D_mul_list[j],ori,velocity1)
            if i=="E":
                np_E=np.asarray(E)
                np_E_trans=np.transpose(np_E)
                one=np.ones((np_E.shape[0],), dtype=int)
                np_E_trans=np.vstack((np_E_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                E_mul=np.matmul(matrix,np_E_trans)
                E_mul_list=((np.transpose(E_mul))[:,0:3]).tolist()
                for j in range(len(E_mul_list)):
                    success &= example_cartesian_action_movement(base,E_mul_list[j],ori,velocity1)
            if i=="F":
                np_F=np.asarray(F)
                np_F_trans=np.transpose(np_F)
                one=np.ones((np_F.shape[0],), dtype=int)
                np_F_trans=np.vstack((np_F_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                F_mul=np.matmul(matrix,np_F_trans)
                F_mul_list=((np.transpose(F_mul))[:,0:3]).tolist()
                for j in range(len(F_mul_list)):
                    success &= example_cartesian_action_movement(base,F_mul_list[j],ori,velocity1)
            if i=="G":
                np_G=np.asarray(G)
                np_G_trans=np.transpose(np_G)
                one=np.ones((np_G.shape[0],), dtype=int)
                np_G_trans=np.vstack((np_G_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                G_mul=np.matmul(matrix,np_G_trans)
                G_mul_list=((np.transpose(G_mul))[:,0:3]).tolist()
                for j in range(len(G_mul_list)):
                    success &= example_cartesian_action_movement(base,G_mul_list[j],ori,velocity1)
            if i=="H":
                np_H=np.asarray(H)
                np_H_trans=np.transpose(np_H)
                one=np.ones((np_H.shape[0],), dtype=int)
                np_H_trans=np.vstack((np_H_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                H_mul=np.matmul(matrix,np_H_trans)
                H_mul_list=((np.transpose(H_mul))[:,0:3]).tolist()
                for j in range(len(H_mul_list)):
                    success &= example_cartesian_action_movement(base,H_mul_list[j],ori,velocity1)
            if i=="I":
                np_I=np.asarray(I)
                np_I_trans=np.transpose(np_I)
                one=np.ones((np_I.shape[0],), dtype=int)
                np_I_trans=np.vstack((np_I_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                I_mul=np.matmul(matrix,np_I_trans)
                I_mul_list=((np.transpose(I_mul))[:,0:3]).tolist()
                for j in range(len(I_mul_list)):
                    success &= example_cartesian_action_movement(base,I_mul_list[j],ori,velocity1)
            if i=="J":
                np_J=np.asarray(J)
                np_J_trans=np.transpose(np_J)
                one=np.ones((np_J.shape[0],), dtype=int)
                np_J_trans=np.vstack((np_J_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                J_mul=np.matmul(matrix,np_J_trans)
                J_mul_list=((np.transpose(J_mul))[:,0:3]).tolist()
                for j in range(len(A_mul_list)):
                    success &= example_cartesian_action_movement(base,J_mul_list[j],ori,velocity1)
            if i=="K":
                np_K=np.asarray(K)
                np_K_trans=np.transpose(np_K)
                one=np.ones((np_K.shape[0],), dtype=int)
                np_K_trans=np.vstack((np_K_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                K_mul=np.matmul(matrix,np_K_trans)
                K_mul_list=((np.transpose(K_mul))[:,0:3]).tolist()
                for j in range(len(K_mul_list)):
                    success &= example_cartesian_action_movement(base,K_mul_list[j],ori,velocity1)
            if i=="L":
                np_L=np.asarray(L)
                np_L_trans=np.transpose(np_L)
                one=np.ones((np_L.shape[0],), dtype=int)
                np_L_trans=np.vstack((np_L_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                L_mul=np.matmul(matrix,np_L_trans)
                L_mul_list=((np.transpose(L_mul))[:,0:3]).tolist()
                for j in range(len(L_mul_list)):
                    success &= example_cartesian_action_movement(base,L_mul_list[j],ori,velocity1)
            if i=="M":
                np_M=np.asarray(M)
                np_M_trans=np.transpose(np_M)
                one=np.ones((np_M.shape[0],), dtype=int)
                np_M_trans=np.vstack((np_M_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                M_mul=np.matmul(matrix,np_M_trans)
                M_mul_list=((np.transpose(M_mul))[:,0:3]).tolist()
                for j in range(len(M_mul_list)):
                    success &= example_cartesian_action_movement(base,M_mul_list[j],ori,velocity1)
            if i=="N":
                np_N=np.asarray(N)
                np_N_trans=np.transpose(np_N)
                one=np.ones((np_N.shape[0],), dtype=int)
                np_N_trans=np.vstack((np_N_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                N_mul=np.matmul(matrix,np_N_trans)
                N_mul_list=((np.transpose(N_mul))[:,0:3]).tolist()
                for j in range(len(N_mul_list)):
                    success &= example_cartesian_action_movement(base,N_mul_list[j],ori,velocity1)
            if i=="O":
                np_O=np.asarray(O)
                np_O_trans=np.transpose(np_O)
                one=np.ones((np_O.shape[0],), dtype=int)
                np_O_trans=np.vstack((np_O_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                O_mul=np.matmul(matrix,np_O_trans)
                O_mul_list=((np.transpose(O_mul))[:,0:3]).tolist()
                for j in range(len(O_mul_list)):
                    success &= example_cartesian_action_movement(base,O_mul_list[j],ori,velocity1)
            if i=="P":
                np_P=np.asarray(P)
                np_P_trans=np.transpose(np_P)
                one=np.ones((np_P.shape[0],), dtype=int)
                np_P_trans=np.vstack((np_P_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                P_mul=np.matmul(matrix,np_P_trans)
                P_mul_list=((np.transpose(P_mul))[:,0:3]).tolist()
                for j in range(len(A_mul_list)):
                    success &= example_cartesian_action_movement(base,P_mul_list[j],ori,velocity1)
            if i=="Q":
                np_Q=np.asarray(Q)
                np_Q_trans=np.transpose(np_Q)
                one=np.ones((np_Q.shape[0],), dtype=int)
                np_Q_trans=np.vstack((np_Q_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                Q_mul=np.matmul(matrix,np_Q_trans)
                Q_mul_list=((np.transpose(Q_mul))[:,0:3]).tolist()
                for j in range(len(Q_mul_list)):
                    success &= example_cartesian_action_movement(base,Q_mul_list[j],ori,velocity1)
            if i=="R":
                np_R=np.asarray(R)
                np_R_trans=np.transpose(np_R)
                one=np.ones((np_R.shape[0],), dtype=int)
                np_R_trans=np.vstack((np_R_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                R_mul=np.matmul(matrix,np_R_trans)
                R_mul_list=((np.transpose(R_mul))[:,0:3]).tolist()
                for j in range(len(R_mul_list)):
                    success &= example_cartesian_action_movement(base,R_mul_list[j],ori,velocity1)
            if i=="S":
                np_S=np.asarray(S)
                np_S_trans=np.transpose(np_S)
                one=np.ones((np_S.shape[0],), dtype=int)
                np_S_trans=np.vstack((np_S_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                S_mul=np.matmul(matrix,np_S_trans)
                S_mul_list=((np.transpose(S_mul))[:,0:3]).tolist()
                for j in range(len(S_mul_list)):
                    success &= example_cartesian_action_movement(base,S_mul_list[j],ori,velocity1)
            if i=="T":
                np_T=np.asarray(T)
                np_T_trans=np.transpose(np_T)
                one=np.ones((np_T.shape[0],), dtype=int)
                np_T_trans=np.vstack((np_T_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                T_mul=np.matmul(matrix,np_T_trans)
                T_mul_list=((np.transpose(T_mul))[:,0:3]).tolist()
                for j in range(len(T_mul_list)):
                    success &= example_cartesian_action_movement(base,T_mul_list[j],ori,velocity1)
            if i=="U":
                np_U=np.asarray(U)
                np_U_trans=np.transpose(np_U)
                one=np.ones((np_U.shape[0],), dtype=int)
                np_U_trans=np.vstack((np_U_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                U_mul=np.matmul(matrix,np_U_trans)
                U_mul_list=((np.transpose(U_mul))[:,0:3]).tolist()
                for j in range(len(U_mul_list)):
                    success &= example_cartesian_action_movement(base,U_mul_list[j],ori,velocity1)
            if i=="V":
                np_V=np.asarray(V)
                np_V_trans=np.transpose(np_V)
                one=np.ones((np_V.shape[0],), dtype=int)
                np_V_trans=np.vstack((np_V_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                V_mul=np.matmul(matrix,np_V_trans)
                V_mul_list=((np.transpose(V_mul))[:,0:3]).tolist()
                for j in range(len(V_mul_list)):
                    success &= example_cartesian_action_movement(base,V_mul_list[j],ori,velocity1)
            if i=="W":
                np_W=np.asarray(W)
                np_W_trans=np.transpose(np_W)
                one=np.ones((np_W.shape[0],), dtype=int)
                np_W_trans=np.vstack((np_W_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                W_mul=np.matmul(matrix,np_W_trans)
                W_mul_list=((np.transpose(W_mul))[:,0:3]).tolist()
                for j in range(len(W_mul_list)):
                    success &= example_cartesian_action_movement(base,W_mul_list[j],ori,velocity1)
            if i=="X":
                np_X=np.asarray(X)
                np_X_trans=np.transpose(np_X)
                one=np.ones((np_X.shape[0],), dtype=int)
                np_X_trans=np.vstack((np_X_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                X_mul=np.matmul(matrix,np_X_trans)
                X_mul_list=((np.transpose(X_mul))[:,0:3]).tolist()
                for j in range(len(X_mul_list)):
                    success &= example_cartesian_action_movement(base,X_mul_list[j],ori,velocity1)
            if i=="Y":
                np_Y=np.asarray(Y)
                np_Y_trans=np.transpose(np_Y)
                one=np.ones((np_Y.shape[0],), dtype=int)
                np_Y_trans=np.vstack((np_Y_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                Y_mul=np.matmul(matrix,np_Y_trans)
                Y_mul_list=((np.transpose(Y_mul))[:,0:3]).tolist()
                for j in range(len(Y_mul_list)):
                    success &= example_cartesian_action_movement(base,Y_mul_list[j],ori,velocity1)
            if i=="Z":
                np_Z=np.asarray(Z)
                np_Z_trans=np.transpose(np_Z)
                one=np.ones((np_Z.shape[0],), dtype=int)
                np_Z_trans=np.vstack((np_Z_trans,one))
                if p!=0:
                    feedback = base_cyclic.RefreshFeedback()
                    matrix[1,3]=feedback.base.tool_pose_y-0.015
                    matrix[2,3]=(args.distance-np.dot(normal_vector[0],matrix[0,3])-np.dot(normal_vector[1],matrix[1,3]))/normal_vector[2]
                Z_mul=np.matmul(matrix,np_Z_trans)
                Z_mul_list=((np.transpose(Z_mul))[:,0:3]).tolist()
                for j in range(len(Z_mul_list)):
                    success &= example_cartesian_action_movement(base,Z_mul_list[j],ori,velocity1)
                    
        #reading the marker placing csv file
        read_csv(Filename_Placing, marker_placement_angle,marker_placement_position,marker_placement_orientation,marker_placement_gripper_position,marker_placement_translation_speed,marker_placement_action_sequence)
        
         #performing the marker placement action
        for i in range(len(marker_placement_action_sequence)):
            if marker_placement_action_sequence[i]==7:
                success &= example_angular_action_movement_1(base,marker_placement_angle[new_angle])
                new_angle=new_angle+1
            elif marker_placement_action_sequence[i]==6:
                success &=example_cartesian_action_movement(base,marker_placement_position[new_position],marker_placement_orientation[new_orientation],marker_placement_translation_speed[speedcounter])
                new_orientation=new_orientation+1
                new_position=new_position+1
                speedcounter=speedcounter+1
            else:
                finger.value= marker_placement_gripper_position[gripper_sequence]
                base.SendGripperCommand(gripper_command)
                time.sleep(1)
                gripper_sequence=gripper_sequence+1

        #setting all counter to zero for reusing
        new_angle=new_position=new_orientation=gripper_sequence=speedcounter=0
        
        return 0 if success else 1
if __name__ == "__main__":
    exit(main())
