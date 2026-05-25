from pathlib import Path
from abc import ABC, abstractmethod

from logger import setup_logger

class User:
    def __init__(self, storage: Path, user_id: int, peer_id: int):
        self.storage = storage
        self.user_id = user_id
        self.peer_id = peer_id
        self.logger = setup_logger(self.__class__.__name__)

    # 임시 저장소 경로 등록
    @property
    def storage(self):
        return self._storage
    
    @storage.setter
    def storage(self, value: Path):
        if not value: 
            raise FileNotFoundError(f"No such path: {value}")
        self._storage = value

    # user_id 등록(자신)
    @property
    def user_id(self):
        return self._user_id
    
    @user_id.setter
    def user_id(self, value: int):
        if value not in range(0, 4): 
            raise ValueError(f"User_id must be in [0..3]: {value}")
        self._user_id = value

    # target_id 등록(송신자 id)
    @property
    def peer_id(self):
        return self._peer_id
    
    @peer_id.setter
    def peer_id(self, value: int):
        if value not in range(0, 4): 
            raise ValueError(f"Peer_id must be in [0..3]: {value}")
        self._peer_id = value
    
    # 서버 ip 연결
    @property
    def server_ip(self):
        return self._server_ip
    
    @server_ip.setter
    def server_ip(self, value: str):
        temp = value.split(":")
        if len(temp) != 2: 
            raise ValueError(f"server_ip must be format x.x.x.x:port. x is integer: {value}")
        ip, port = temp[0], temp[1]
        if not port.isdigit():
            raise ValueError(f"port must be integer")

        numbers = ip.split(".")
        if len(numbers) != 4: 
            raise ValueError(f"server_ip must be format x.x.x.x:port. x is integer: {value}")
        for number in numbers:
            if not number.isdigit():
                raise ValueError(f"server_ip must be format x.x.x.x:port. x is integer: {value}")
            if not 0 <= int(number) <= 255:
                raise ValueError(f"server_ip number must be between 0 and 255: {value}")
                                             
        self._server_ip = value

    # 서버 커넥션 연결 로직
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.info(f"{self.__class__.__name__} is removed. Start storage cleanup")
        for file in self.storage.glob("*.json"):
            file.unlink()
            self.logger.info(f"{file} is removed")
        