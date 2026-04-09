"""
lerobot-remote-module
=====================
LeRobot 원격 훈련 및 추론 모듈.

데스크탑 (RTX 3060):
    from module.desktop import RobotConnector, DataCollector, InferenceClient, TrainingClient

서버 (GPU):
    from module.server import LeRobotGRPCServer

모델 플러그인:
    from module.utils import ModelRegistry
    ModelRegistry.auto_discover()
    model = ModelRegistry.build("act", config={...})
"""

__version__ = "0.1.0"
