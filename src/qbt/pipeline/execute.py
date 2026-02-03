from qbt.execution.alpaca_client import AlpacaTradingAPI
from qbt.execution.rebalancing import *




def place_orders(client, orders):

    for asset, delta_qty in orders:
        if delta_qty > 0:
            side = "buy"
            position_intent = "buy_to_open"
        elif delta_qty < 0:
            side = "sell"
            position_intent = "sell_to_close"

        client.place_order()
    pass


def execute_weights():

    # get target weights

    # get current weights

    # get change amount 

    # execute  


    pass

if __name__ == '__main__': 


    cfg = {}
    client = AlpacaTradingAPI(cfg=cfg)

    pos = client.get_active_positions()

    print(pos)

    equity = client.get_equity()
    print(equity)