import sys
import os
import time
import threading
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
        
        vect1=[]
        vect2=[]
        vect3=[]
        start_position=[]
        thetax=0
        thetay=0
        thetaz=0
        
        
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
        new_angle=new_position=new_orientation=gripper_sequence=speedcounter=0
        

        print("Use the controller to touch 3 point on the surface with marker.")
        input("Take the robot in the to the 1st Point.(Top left) Press enter when it is done")
        feedback = base_cyclic.RefreshFeedback()
        vect1.append(feedback.base.tool_pose_x)
        vect1.append(feedback.base.tool_pose_y)
        vect1.append(feedback.base.tool_pose_z)
        thetax=thetax+feedback.base.tool_pose_theta_x
        thetay=thetay+feedback.base.tool_pose_theta_y
        thetaz=thetaz+feedback.base.tool_pose_theta_z
        input("Take the robot in the to the 2nd Point.(Bottom left) Press enter when it is done")
        feedback = base_cyclic.RefreshFeedback()
        vect2.append(feedback.base.tool_pose_x)
        vect2.append(feedback.base.tool_pose_y)
        vect2.append(feedback.base.tool_pose_z)
        thetax=thetax+feedback.base.tool_pose_theta_x
        thetay=thetay+feedback.base.tool_pose_theta_y
        thetaz=thetaz+feedback.base.tool_pose_theta_z
        input("Take the robot in the to the 3rd Point.(Bottom right) Press enter when it is done")
        feedback = base_cyclic.RefreshFeedback()
        vect3.append(feedback.base.tool_pose_x)
        vect3.append(feedback.base.tool_pose_y)
        vect3.append(feedback.base.tool_pose_z)
        thetax=thetax+feedback.base.tool_pose_theta_x
        thetay=thetay+feedback.base.tool_pose_theta_y
        thetaz=thetaz+feedback.base.tool_pose_theta_z
        input("Take the robot to the starting point for writing letter. Press enter when it is done")
        feedback = base_cyclic.RefreshFeedback()
        start_position.append(feedback.base.tool_pose_x)
        start_position.append(feedback.base.tool_pose_y)
        start_position.append(feedback.base.tool_pose_z)
        
        
        #Vector Math
        np_vect1=np.asarray(vect1)
        np_vect2=np.asarray(vect2)
        np_vect3=np.asarray(vect3)
        
        np_vect21=np_vect1-np_vect2 #x-directional vector
        np_vect23=-(np_vect3-np_vect2)#y-directional vector
        
        normal_vector=np.cross(np_vect21,np_vect23)
        
        unit_x= np_vect21/ np.linalg.norm(np_vect21)#unit vector in x direction
        unit_z=normal_vector/np.linalg.norm(normal_vector)#unit vector in z direction
        unit_y=np.cross(unit_z,unit_x)#unit vector in y direction
        
        vect1=unit_x.tolist()
        vect2=unit_y.tolist()
        vect3=unit_z.tolist()
        
        distance=np.dot(normal_vector,np_vect1)#distance use to create the equation of plane
        const_orientation=[thetax/3,thetay/3,thetaz/3]#averge angles to get the orientation
        print("vect1 ", vect1)
        print("vect2 ", vect2)
        print("vect3 ", vect3)
        print("normal-vector ", normal_vector.tolist())
        print("distance ", distance)
        print("start_position ",start_position)
        print("orientation ", const_orientation)
        
        print("Allow the robot to restablish session with the pc. DO not move the robot with the controller.")
        time.sleep(10)
        
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
        #command for running the face recognitation code
        cmd="python3 /home/dospina/New_FR/Face_Recognition.py --vect1 "+ str(vect1[0])+" "+ str(vect1[1])+" "+str(vect1[2])+" --vect2 "+str(vect2[0])+" "+str(vect2[1])+" "+str(vect2[2])+" "+ " --vect3 "+str(vect3[0])+" "+str(vect3[1])+" "+str(vect3[2])+" --orientation "+str(const_orientation[0])+" "+str(const_orientation[1])+" "+str(const_orientation[2])+" --start_position "+str(start_position[0])+" "+str(start_position[1])+" "+str(start_position[2])+ " --normal_vector "+str((normal_vector.tolist())[0])+" "+str((normal_vector.tolist())[1])+" "+str((normal_vector.tolist())[2])+" -distance "+str(distance)
        os.system(cmd)
        
        return 0 if success else 1
if __name__ == "__main__":
    exit(main())
