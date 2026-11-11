"""Damped least-squares IK for the G1 left arm (data-collection tooling only).

Uses the simulator's kinematic model and state, so it is privileged and must
never be called by the evaluated policy. It turns a hand target (pose in the
pelvis frame) into the 7 absolute left-arm joint targets of the action vector.
"""

import mujoco
import numpy as np

PREFIX = "g1_29dof_with_dex1_1/"
LEFT_ARM = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
EE_SITE = PREFIX + "left_end_effector"
PELVIS = PREFIX + "pelvis"


class LeftArmIK:
    def __init__(self, model, rot_weight=0.3, damping=0.05):
        self.m = model
        self.d = mujoco.MjData(model)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + n) for n in LEFT_ARM]
        if min(jids) < 0:
            raise RuntimeError("left-arm joint not found")
        self.qadr = model.jnt_qposadr[jids]
        self.dadr = model.jnt_dofadr[jids]
        self.lo, self.hi = model.jnt_range[jids].T
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        self.pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PELVIS)
        if self.site < 0 or self.pelvis < 0:
            raise RuntimeError("end-effector site or pelvis body not found")
        self.rot_weight, self.damping = rot_weight, damping
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self.bias = np.zeros(3)  # integral term: compensates PD sag under gravity/load

    def track(self, data, q_cmd, pos, mat, gain=0.03, max_bias=0.06, window=0.03):
        """IK on the commanded joints, nudged by the measured hand error (world pos/mat).

        The sag-compensation integral only accumulates while the hand is close to
        the target (anti-windup): when contact or reach blocks the hand, it decays
        instead of pushing harder, which would make the arm chatter.
        """
        err = pos - data.site_xpos[self.site]
        if np.linalg.norm(err) < window:
            self.bias = np.clip(self.bias + gain * err, -max_bias, max_bias)
        else:
            self.bias *= 0.98
        return self.solve(data.qpos.copy(), q_cmd, pos + self.bias, mat)

    # --- frames -----------------------------------------------------------------
    @staticmethod
    def pelvis_frame(data, body):
        return data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()

    def to_local(self, data, pos, mat):
        p, r = self.pelvis_frame(data, self.pelvis)
        return r.T @ (pos - p), r.T @ mat

    def to_world(self, data, pos, mat):
        p, r = self.pelvis_frame(data, self.pelvis)
        return p + r @ pos, r @ mat

    # --- kinematics -------------------------------------------------------------
    def fk(self, qpos, q):
        self.d.qpos[:] = qpos
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, qpos, q0, pos, mat, iters=15):
        """Joint targets near q0 whose FK (with the rest of qpos fixed) hits pos/mat (world)."""
        q = np.clip(np.asarray(q0, np.float64).copy(), self.lo, self.hi)
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, np.asarray(mat, np.float64).ravel())
        cur_quat, err_r = np.zeros(4), np.zeros(3)
        for _ in range(iters):
            p, r = self.fk(qpos, q)
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mju_mat2Quat(cur_quat, r.ravel())
            mujoco.mju_subQuat(err_r, target_quat, cur_quat)  # in the current site frame
            err = np.concatenate([pos - p, self.rot_weight * (r @ err_r)])
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jacSite(self.m, self.d, self._jacp, self._jacr, self.site)
            jac = np.vstack([self._jacp[:, self.dadr], self.rot_weight * self._jacr[:, self.dadr]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + self.damping**2 * np.eye(6), err)
            q = np.clip(q + dq, self.lo, self.hi)
        return q
