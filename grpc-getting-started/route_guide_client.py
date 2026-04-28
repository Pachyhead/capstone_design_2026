"""The Python implementation of the gRPC route guide client."""

import logging

import grpc

import route_guide_pb2
import route_guide_pb2_grpc

import os
from dotenv import load_dotenv

def format_point(point):
    # Not delegating in point.__str__ because it is an empty string when its
    # values are zero. In addition, it puts a newline between the fields.
    return f"latitude: {point.latitude}, longitude: {point.longitude}"


def run():
    """
    Codelab Hint: Logic for your gRPC Client will be added here.
    Steps include:
     1. Create a connection to the gRPC server using grpc.insecure_channel()
     2. Call service methods on the client to interact with the server.
    """
    load_dotenv() # load .env file's variables to os.environ
    server_addr = f"{os.environ.get('SERV_IP', '')}:{os.environ.get('SERV_PORT', '')}"
    channel = grpc.insecure_channel(server_addr) # RouteGuideStub 인스턴스화
    stub = route_guide_pb2_grpc.RouteGuideStub(channel)
    
    point = route_guide_pb2.Point(latitude=412346009, longitude=-744026814) # 서비스를 호출할 Point 정의
    feature = stub.GetFeature(point) # 서버 메서드 호출(rpc 호출)
    print(feature)

    if feature.name:
        print(f"Feature called '{feature.name}' at {format_point(feature.location)}")
    else:
        printf(f"Found no feature at {format_point(feature.location)}")    


if __name__ == "__main__":
    logging.basicConfig()
    run()
