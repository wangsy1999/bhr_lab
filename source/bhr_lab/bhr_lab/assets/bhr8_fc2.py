import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from bhr_lab.config import PROJECT_ROOT

DEG2RAD = 3.14159265358979323846 / 180.0

BHR8_FC2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJECT_ROOT}/robots/bhr/bhr8_fc2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # ! disable self collisions in fisrt training
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.92), 
        joint_pos={
            ".*hipYaw"  :  0.0,
            ".*hipRoll" :  0.0,
            ".*hipPitch": -15.0*DEG2RAD,
            ".*knee"    :  30.0*DEG2RAD,
            ".*ankle1"  : -15.0*DEG2RAD,
            ".*ankle2"  :  0.0,
            ".*shoulderPitch":  0.0,
            "lshoulderRoll" : 5.0*DEG2RAD,
            "rshoulderRoll" :-5.0*DEG2RAD,
            ".*shoulderYaw"  :  0.0,
            ".*elbow"        :-30.0*DEG2RAD,    
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip.*", ".*knee"],
            effort_limit_sim={
                ".*hipYaw"  : 60.0, 
                ".*hipRoll" : 60.0, 
                ".*hipPitch": 80.0, 
                ".*knee"    : 100.0, 
            },
            velocity_limit_sim={
                ".*hipYaw"  : 30.0, 
                ".*hipRoll" : 30.0, 
                ".*hipPitch": 30.0, 
                ".*knee"    : 30.0, 
            },
            stiffness={
                ".*hipYaw"  : 150.0, 
                ".*hipRoll" : 150.0, 
                ".*hipPitch": 200.0, 
                ".*knee"    : 200.0, 
            },
            damping={
                ".*hipYaw"  : 2.5, 
                ".*hipRoll" : 2.5, 
                ".*hipPitch": 2.5, 
                ".*knee"    : 2.5, 
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle.*"],
            effort_limit_sim={
                ".*ankle1": 80.0, 
                ".*ankle2": 60.0,
            },
            velocity_limit_sim={
                ".*ankle1": 30.0, 
                ".*ankle2": 30.0,
            },
            stiffness={
                ".*ankle1": 40.0, 
                ".*ankle2": 40.0,
            },
            damping={
                ".*ankle1": 2.0, 
                ".*ankle2": 2.0,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*shoulder.*", ".*elbow"],
            effort_limit_sim={
                ".*shoulderPitch": 100.0, 
                ".*shoulderRoll" : 100.0, 
                ".*shoulderYaw"  : 100.0,
                ".*elbow"        : 100.0,
            },
            velocity_limit_sim={
                ".*shoulderPitch": 100.0, 
                ".*shoulderRoll" : 100.0, 
                ".*shoulderYaw"  : 100.0,
                ".*elbow"        : 100.0,
            },
            stiffness={
                ".*shoulderPitch": 100.0, 
                ".*shoulderRoll" : 100.0, 
                ".*shoulderYaw"  : 100.0,
                ".*elbow"        : 100.0,
            },
            damping={
                ".*shoulderPitch": 2.0, 
                ".*shoulderRoll" : 2.0, 
                ".*shoulderYaw"  : 2.0,
                ".*elbow"        : 2.0,
            },
        ),
    },
)

BHR8_FC2_NOARM_CFG = BHR8_FC2_CFG.copy()  # type: ignore
BHR8_FC2_NOARM_CFG.spawn.usd_path = f"{PROJECT_ROOT}/robots/bhr/bhr8_fc2/bhr8_fc2_noarm_lim.usd"
BHR8_FC2_NOARM_CFG.actuators.pop("arms")
BHR8_FC2_NOARM_CFG.init_state.joint_pos.pop(".*shoulderPitch")
BHR8_FC2_NOARM_CFG.init_state.joint_pos.pop("lshoulderRoll" )
BHR8_FC2_NOARM_CFG.init_state.joint_pos.pop("rshoulderRoll" )
BHR8_FC2_NOARM_CFG.init_state.joint_pos.pop(".*shoulderYaw"  )
BHR8_FC2_NOARM_CFG.init_state.joint_pos.pop(".*elbow"        )

BHR8_FC2_NAY_CFG = BHR8_FC2_CFG.copy()  # type: ignore
BHR8_FC2_NAY_CFG.spawn.usd_path = f"{PROJECT_ROOT}/robots/bhr/bhr8_fc2/bhr8_fc2_nay.usd"
BHR8_FC2_NAY_CFG.actuators["arms"].effort_limit_sim.pop(".*shoulderYaw")
BHR8_FC2_NAY_CFG.actuators["arms"].velocity_limit_sim.pop(".*shoulderYaw")
BHR8_FC2_NAY_CFG.actuators["arms"].stiffness.pop(".*shoulderYaw")
BHR8_FC2_NAY_CFG.actuators["arms"].damping.pop(".*shoulderYaw")
BHR8_FC2_NAY_CFG.init_state.joint_pos.pop(".*shoulderYaw"  )

"""Configuration for the BHR8_FC2 Humanoid robot."""
