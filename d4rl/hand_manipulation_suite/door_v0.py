import numpy as np
from gym import utils
from gym import spaces
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
from d4rl import offline_env
import os

from math import radians

ADD_BONUS_REWARDS = True

class DoorEnvV0(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_door.xml', 5)
        
        # Override action_space to -1, 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape)
        
        # change actuator sensitivity
        
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        utils.EzPickle.__init__(self)
        ob = self.reset_model()
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.door_hinge_did = self.model.jnt_dofadr[self.model.joint_name2id('door_hinge')]
        self.grasp_sid = self.model.site_name2id('S_grasp')
        self.handle_sid = self.model.site_name2id('S_handle')
        self.door_bid = self.model.body_name2id('frame')

    def step(self, qp):
        '''
        a = np.clip(a, -1.0, 1.0)
        
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
            print("act_mid {} act_rng {}".format(self.act_mid, self.act_rng))
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        '''
        
        state_dict = self.get_env_state()
        state_dict['door_body_pos'] = [0.0] * 3
        #print('qp', qp)
        #print('pos', state_dict['door_body_pos'])
        #qp = state_dict['qpos']
        #qv = state_dict['qvel']
        '''
        qp = np.array([0.0] * 30)
        qv = np.array([0.0] * 30)

        # indnex finger hardcode TEST
        move_no = 8
        qp = self.finger_mvt(qp, move_no)
        
        self.set_state(qp, qv)
        '''
        self.set_env_state(state_dict)
        #qp = state_dict['qpos']
        
        self.do_simulation(qp, self.frame_skip)

        ob = self.get_obs()

        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]

        #print('here', state_dict['door_body_pos'])

        # get to handle
        reward = -0.1*np.linalg.norm(palm_pos-handle_pos)
        # open door
        reward += -0.1*(door_pos - 1.57)*(door_pos - 1.57)
        # velocity cost
        reward += -1e-5*np.sum(self.data.qvel**2)

        if ADD_BONUS_REWARDS:
            # Bonus
            if door_pos > 0.2:
                reward += 2
            if door_pos > 1.0:
                reward += 8
            if door_pos > 1.35:
                reward += 10

        goal_achieved = True if door_pos >= 1.35 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)
        #return qp

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = np.array([self.data.qpos[self.door_hinge_did]])
        if door_pos > 1.0:
            door_open = 1.0
        else:
            door_open = -1.0
        latch_pos = qp[-1]
        #print(qp)
        return np.concatenate([qp[1:-2], [latch_pos], door_pos, palm_pos, handle_pos, palm_pos-handle_pos, [door_open]])

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        #qp[11] = 0.45
        self.set_state(qp, qv)
        '''
        self.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=-0.3, high=-0.2)
        self.model.body_pos[self.door_bid,1] = self.np_random.uniform(low=0.25, high=0.35)
        self.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=0.252, high=0.35)
        '''
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos)

    def finger_mvt(self, qp, move_no):
        qp[2] = 1.57 # hand positioned sideways
        #qp[3] = 3.14 # hand faces upwards
        if move_no == 1:
            qp[7] = radians(136-117)
            qp[8] = radians(171-95)
            qp[9] = radians(140-65)
        elif move_no == 2:
            qp[7] = radians(57-117)
            qp[8] = radians(20-95)
            qp[9] = radians(102-65)
        elif move_no == 8:
            qp[7] = radians(10)
            qp[8] = radians(10)
            qp[9] = radians(-3)
        

        return qp

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        '''
        # This section has been moved to step()
        #qp = state_dict['qpos']
        #qv = state_dict['qvel']
        
        qp = np.array([0.0] * 30)
        qv = np.array([0.0] * 30)

        # indnex finger hardcode TEST
        move_no = 8
        qp = self.finger_mvt(qp, move_no)
        
        self.set_state(qp, qv)
        '''
        self.model.body_pos[self.door_bid] = state_dict['door_body_pos']
        
        #print("qp", qp)
        '''
        print("qp", qp)
        print("qv", qv)
        print("pos", self.model.body_pos)
        '''
        self.sim.forward()
        #return qp

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        from mujoco_py.generated import const
        self.viewer.cam.type = const.CAMERA_FIXED
        self.viewer.cam.fixedcamid = 0
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5
        #self.viewer._record_video = True
        #self.viewer._video_path = "/output.mp4"

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if door open for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
