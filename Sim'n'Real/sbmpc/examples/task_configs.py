"""
纯MPPI配置 - 不依赖反馈增益的优化参数
"""
import jax.numpy as jnp

class TaskConfig:
    @staticmethod
    def get_config_for_task(task_name, step=1):
        """根据任务和步骤返回配置"""
        
        if task_name == "hover":
            return TaskConfig._hover_config(step)
        elif task_name == "reach_point":
            return TaskConfig._reach_point_config(step)
        elif task_name == "arm_control":
            return TaskConfig._arm_control_config(step)
        elif task_name == "trajectory":
            return TaskConfig._trajectory_config(step)
        elif task_name == "end_effector_trajectory":
            return TaskConfig._end_effector_trajectory_config(step)
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    @staticmethod
    def _hover_config(step):
        """悬停任务配置（纯MPPI优化）"""
        if step == 1:
            return {
                'dt': 0.02,
                'horizon': 30,      # 适中的预测时域
                'samples': 2000,    # 足够的采样数
                'lambda': 1.0,      # 标准lambda值（关键改变）
                'noise': jnp.array([
                    1.0,    # 适中的推力噪声
                    0.15,   # 适中的扭矩噪声
                    0.15,    
                    0.15,    
                    0.0,    
                    0.0     
                ])
            }
        elif step == 2:
            return {
                'dt': 0.02,
                'horizon': 30,
                'samples': 2000,
                'lambda': 1.0,
                'noise': jnp.array([
                    0.8,    
                    0.12,   
                    0.12,   
                    0.12,   
                    0.05,   
                    0.05    
                ])
            }
        else:  # step 3
            return {
                'dt': 0.02,
                'horizon': 35,
                'samples': 2500,
                'lambda': 0.8,
                'noise': jnp.array([
                    0.9,
                    0.13,
                    0.13,
                    0.13,
                    0.06,
                    0.06
                ])
            }
    
    @staticmethod
    def _reach_point_config(step):
        """到达目标点配置"""
        return {
            'dt': 0.02,
            'horizon': 35,
            'samples': 2500,
            'lambda': 0.5,
            'noise': jnp.array([
                2.0,    
                0.2,
                0.2,
                0.2,
                0.0,
                0.0
            ])
        }
    
    @staticmethod
    def _arm_control_config(step):
        """机械臂控制配置"""
        return {
            'dt': 0.02,
            'horizon': 40,
            'samples': 2000,
            'lambda': 0.6,
            'noise': jnp.array([
                1.0,    # 增大推力噪声，提高探索能力
                0.15,   # 增大扭矩噪声
                0.15,
                0.15,
                0.4,    # 显著增大关节噪声
                0.4
            ])
        }
    
    @staticmethod  
    def _trajectory_config(step):
        """轨迹跟踪配置"""
        return {
            'dt': 0.02,
            'horizon': 35,
            'samples': 2500,
            'lambda': 0.5,
            'noise': jnp.array([
                1.2,
                0.15,
                0.15,
                0.15,
                0.1,
                0.1
            ])
        }
    
    @staticmethod
    def _end_effector_trajectory_config(step):
        """末端执行器轨迹跟踪配置（借鉴arm_control的成功参数）"""
        if step == 1:
            return {
                'dt': 0.02,
                'horizon': 40,      # 借鉴arm_control
                'samples': 2000,    # 借鉴arm_control
                'lambda': 0.6,      # 借鉴arm_control
                'noise': jnp.array([
                    1.0,    # 借鉴arm_control的推力噪声
                    0.15,   # 借鉴arm_control的扭矩噪声
                    0.15,
                    0.15,
                    0.0,    # step1不控制关节
                    0.0
                ])
            }
        else:  
            return {
                'dt': 0.04,
                'horizon': 30,      
                'samples': 1200,    
                'lambda': 0.7,      
                'noise': jnp.array([
                    1.6,    # 减少推力噪声
                    0.6,    # 减少X扭矩（避免过冲）
                    0.7,    # 保持Y
                    0.3,    # Z很好
                    1.0,    # 平衡的关节噪声
                    0.8   
                ])
            }