from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import Parameters
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import GetParametersIns
from flwr.server.utils.tensorboard import tensorboard
from flwr.server.strategy import FedAvg, FedAdam
import argparse

import requests
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", help="modified model name", type=str, nargs='?')
parser.add_argument("-p", "--port", help="local port", type=str, default="8080")
args = parser.parse_args()

def main():
    total_rounds = 2
    clients = 2
    # Decorated strategy
    strategy = FedAvg(min_fit_clients=clients,
                      min_evaluate_clients=clients,
                      min_available_clients=clients,
                      )

    send_line('Starting experiment')
    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:" + args.port,
        config=fl.server.ServerConfig(num_rounds=total_rounds),
        strategy=strategy,
    )
    send_line('Experiment done !!!!!')
    sys.exit()
    
def send_line(message:str):
    token = '7ZPjzeQrRcI70yDFnhBd4A6xpU8MddE7MntCSdbLBgC'
    url = 'https://notify-api.line.me/api/notify'
    headers = {
        'Authorization': f'Bearer {token}'
    }
    data = {
        'message':message
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print("LINE message send sucessfuly")
    else:
        print("LINE message send error：", response.status_code)
    
if __name__ == "__main__":
    main()
    
