from collections import deque
import time
import random
from abc import ABC

from enum import Enum


class OrderType(Enum):
    LIMIT = 1
    MARKET = 2
    IOC = 3


class OrderSide(Enum):
    BUY = 1
    SELL = 2


class NonPositiveQuantity(Exception):
    pass


class NonPositivePrice(Exception):
    pass


class InvalidSide(Exception):
    pass


class UndefinedOrderType(Exception):
    pass


class UndefinedOrderSide(Exception):
    pass


class NewQuantityNotSmaller(Exception):
    pass


class UndefinedTraderAction(Exception):
    pass


class UndefinedResponse(Exception):
    pass


class Order(ABC):
    def __init__(self, id, symbol, quantity, side, time):
        self.id = id
        self.symbol = symbol
        if quantity > 0:
            self.quantity = quantity
        else:
            raise NonPositiveQuantity("Quantity Must Be Positive!")
        if side in [OrderSide.BUY, OrderSide.SELL]:
            self.side = side
        else:
            raise InvalidSide("Side Must Be Either \"Buy\" or \"OrderSide.SELL\"!")
        self.time = time


class LimitOrder(Order):
    def __init__(self, id, symbol, quantity, price, side, time):
        super().__init__(id, symbol, quantity, side, time)
        if price > 0:
            self.price = price
        else:
            raise NonPositivePrice("Price Must Be Positive!")
        self.type = OrderType.LIMIT


class MarketOrder(Order):
    def __init__(self, id, symbol, quantity, side, time):
        super().__init__(id, symbol, quantity, side, time)
        self.type = OrderType.MARKET


class IOCOrder(Order):
    def __init__(self, id, symbol, quantity, price, side, time):
        super().__init__(id, symbol, quantity, side, time)
        if price > 0:
            self.price = price
        else:
            raise NonPositivePrice("Price Must Be Positive!")
        self.type = OrderType.IOC


class FilledOrder(Order):
    def __init__(self, id, symbol, quantity, price, side, time, limit=False):
        super().__init__(id, symbol, quantity, side, time)
        self.price = price
        self.limit = limit


trader_to_exchange = deque()
exchange_to_trader = [deque() for _ in range(100)]


# Above you are given two deques where the orders submitted to the exchange and back to the trader
# are expected to be populated by the trading exchange simulator
# The first is trader_to_exchange, a deque of orders to be populated for the exchange to execute
# The second is a list of 100 deques exchange_to_trader, which are acknowledgements from the exchange
# to each of the 100 traders for trades executed on their behalf

# Below you have an implementation of a simulated thread to be used where each trader is a separate thread
class MyThread:
    list_of_threads = []

    def __init__(self, id='NoID'):
        MyThread.list_of_threads.append(self)
        self.is_started = False
        self.id = id

    def start(self):
        self.is_started = True

    def join(self):
        #if self.list_of_threads is not None: return
        #print('Trader ' + str(self.id) + ' will be waited')
        pass


# Paste in your implementation for the matching engine below

# ----------------------------------------------------------
# PASTE MATCHING ENGINE FROM Q2 HERE
class MatchingEngine():
    def __init__(self):
        self.bid_book = []
        self.ask_book = []
        # These are the order books you are given and expected to use for matching the orders below

    # Note: As you implement the following functions keep in mind that these enums are available:
    #     class OrderType(Enum):
    #         LIMIT = 1
    #         MARKET = 2
    #         IOC = 3

    #     class OrderSide(Enum):
    #         BUY = 1
    #         SELL = 2

    def handle_order(self, order):
        # Implement this function
        # In this function you need to call different functions from the matching engine
        # depending on the type of order you are given
        if order.type == OrderType.LIMIT:
            return self.handle_limit_order(order)
        elif order.type == OrderType.MARKET:
            return self.handle_market_order(order)
        elif order.type == OrderType.IOC:
            return self.handle_ioc_order(order)
        # You need to raise the following error if the type of order is ambiguous
        else:
            raise UndefinedOrderType("Undefined Order Type!")

    def handle_limit_order(self, order):
        # Implement this function
        # Keep in mind what happens to the orders in the limit order books when orders get filled
        # or if there are no crosses from this order
        # in other words, handle_limit_order accepts an arbitrary limit order that can either be
        # filled if the limit order price crosses the book, or placed in the book. If the latter,
        # pass the order to insert_limit_order below.
        filled_orders = []
        # The orders that are filled from the market order need to be inserted into the above list
        if order.side == OrderSide.BUY:
            if len(self.ask_book) == 0:
                # filled_orders.append(order)
                # self.bid_book.append(order)
                self.insert_limit_order(order)
            else:
                while order.quantity > 0 and len(self.ask_book) > 0 and order.price > self.ask_book[0].price:
                    if order.quantity == self.ask_book[0].quantity:
                        filled_orders.append(self.ask_book[0])
                        del (self.ask_book[0])
                    elif order.quantity > self.ask_book[0].quantity:
                        filled_orders.append(self.ask_book[0])
                        order.quantity -= self.ask_book[0].quantity
                        del (self.ask_book[0])
                    elif order.quantity < self.ask_book[0].quantity:
                        filled_orders.append(order)
                        self.ask_book[0].quantity -= order.quantity
                        filled_orders.append(self.ask_book[0])
                        order.quantity = 0
                if order.quantity > 0:
                    self.insert_limit_order(order)
                return filled_orders


        elif order.side == OrderSide.SELL:
            if len(self.bid_book) == 0:
                # filled_orders.append(order)
                # self.ask_book.append(order)
                self.insert_limit_order(order)
            else:
                while order.quantity > 0 and len(self.bid_book) > 0 and order.price < self.bid_book[0].price:
                    if order.quantity == self.bid_book[0].quantity:
                        filled_orders.append(self.bid_book[0])
                        del (self.bid_book[0])
                    elif order.quantity > self.bid_book[0].quantity:
                        filled_orders.append(self.bid_book[0])
                        order.quantity -= self.bid_book[0].quantity
                        del (self.bid_book[0])
                    elif order.quantity < self.bid_book[0].quantity:
                        filled_orders.append(order)
                        self.bid_book[0].quantity -= order.quantity
                        filled_orders.append(self.bid_book[0])
                        order.quantity = 0
                if order.quantity > 0:
                    self.insert_limit_order(order)
                return filled_orders
        else:
            raise UndefinedOrderSide("Undefined Order Side!")

        # The filled orders are expected to be the return variable (list)
        return filled_orders

        # You need to raise the following error if the side the order is for is ambiguous

    def handle_market_order(self, order):
        # Implement this function
        filled_orders = []
        # The orders that are filled from the market order need to be inserted into the above list
        if order.side == OrderSide.BUY:
            while order.quantity > 0 and len(self.ask_book) > 0:
                if order.quantity == self.ask_book[0].quantity:
                    filled_orders.append(self.ask_book[0])
                    del (self.ask_book[0])
                elif order.quantity > self.ask_book[0].quantity:
                    filled_orders.append(self.ask_book[0])
                    order.quantity -= self.ask_book[0].quantity
                    del (self.ask_book[0])
                elif order.quantity < self.ask_book[0].quantity:
                    self.ask_book[0].quantity -= order.quantity
                    filled_orders.append(self.ask_book[0])
                    order.quantity = 0
            return filled_orders

        elif order.side == OrderSide.SELL:
            while order.quantity > 0 and len(self.bid_book) > 0:
                if order.quantity == self.bid_book[0].quantity:
                    filled_orders.append(self.bid_book[0])
                    del (self.bid_book[0])
                elif order.quantity > self.bid_book[0].quantity:
                    filled_orders.append(self.bid_book[0])
                    order.quantity -= self.bid_book[0].quantity
                    del (self.bid_book[0])
                elif order.quantity < self.bid_book[0].quantity:
                    self.bid_book[0].quantity -= order.quantity
                    filled_orders.append(self.bid_book[0])
                    order.quantity = 0
            return filled_orders
        else:
            # The filled orders are expected to be the return variable (list)

            # You need to raise the following error if the side the order is for is ambiguous
            raise UndefinedOrderSide("Undefined Order Side!")

    def handle_ioc_order(self, order):
        # Implement this function
        filled_orders = []
        # The orders that are filled from the ioc order need to be inserted into the above list
        if order.side == OrderSide.BUY:
            if len(self.ask_book) == 0:
                return filled_orders
            if order.quantity == self.ask_book[0].quantity and order.price >= self.ask_book[0].price:
                filled_orders.append(self.ask_book[0])
                del (self.ask_book[0])
            elif order.quantity < self.ask_book[0].quantity and order.price >= self.ask_book[0].price:
                self.ask_book[0].quantity -= order.quantity
                filled_orders.append(self.ask_book[0])
                order.quantity = 0
            return filled_orders

        elif order.side == OrderSide.SELL:
            if len(self.bid_book) == 0:
                return filled_orders
            if order.quantity == self.bid_book[0].quantity and order.price <= self.bid_book[0].price:
                filled_orders.append(self.bid_book[0])
                del (self.bid_book[0])
            elif order.quantity < self.bid_book[0].quantity and order.price <= self.bid_book[0].price:
                self.bid_book[0].quantity -= order.quantity
                filled_orders.append(self.bid_book[0])
                order.quantity = 0
            return filled_orders
        else:
            # The filled orders are expected to be the return variable (list)

            # You need to raise the following error if the side the order is for is ambiguous
            raise UndefinedOrderSide("Undefined Order Side!")

    def insert_limit_order(self, order):
        assert order.type == OrderType.LIMIT
        # Implement this function
        # this function's sole puporse is to place limit orders in the book that are guaranteed
        # to not immediately fill
        if order.side == OrderSide.BUY:
            self.bid_book.append(order)
            self.bid_book = sorted(self.bid_book, key=lambda x: (-x.price, x.time))
        elif order.side == OrderSide.SELL:
            self.ask_book.append(order)
            self.ask_book = sorted(self.ask_book, key=lambda x: (x.price, x.time))
        else:
            # You need to raise the following error if the side the order is for is ambiguous
            raise UndefinedOrderSide("Undefined Order Side!")

    def amend_quantity(self, id, quantity):
        # Implement this function
        # Hint: Remember that there are two order books, one on the bid side and one on the ask side
        bidcheck = next((o for o in self.bid_book if o.id == id), None)
        askcheck = next((o for o in self.ask_book if o.id == id), None)

        if bidcheck != None and bidcheck.quantity > quantity:
            if bidcheck.type == OrderType.LIMIT:
                self.bid_book.insert(0, LimitOrder(id, bidcheck.symbol, quantity, bidcheck.price, bidcheck.side,
                                                   time.time()))
                self.bid_book.remove(bidcheck)
            elif bidcheck.type == OrderType.MARKET:
                self.bid_book.insert(0, MarketOrder(id, bidcheck.symbol, quantity, bidcheck.side, time.time()))
                self.bid_book.remove(bidcheck)
            elif bidcheck.type == OrderType.IOC:
                self.bid_book.insert(0, IOCOrder(id, bidcheck.symbol, quantity, bidcheck.side, time.time()))
                self.bid_book.remove(bidcheck)

        elif askcheck != None and askcheck.quantity > quantity:
            if askcheck.type == OrderType.LIMIT:
                self.ask_book.insert(0, LimitOrder(id, askcheck.symbol, quantity, askcheck.price, askcheck.side,
                                                   time.time()))
                self.ask_book.remove(askcheck)
            elif askcheck.type == OrderType.MARKET:
                self.ask_book.insert(0, MarketOrder(id, askcheck.symbol, quantity, askcheck.side, time.time()))
                self.ask_book.remove(askcheck)
            elif askcheck.type == OrderType.IOC:
                self.ask_book.insert(0, IOCOrder(id, askcheck.symbol, quantity, askcheck.side, time.time()))
                self.ask_book.remove(askcheck)
        # You need to raise the following error if the user attempts to modify an order
        # with a quantity that's greater than given in the existing order
        else:
            raise NewQuantityNotSmaller("Amendment Must Reduce Quantity!")

        return False

    def cancel_order(self, id):
        # Implement this function
        # Think about the changes you need to make in the order book based on the parameters given
        bidcheck = next((o for o in self.bid_book if o.id == id), None)
        askcheck = next((o for o in self.ask_book if o.id == id), None)

        if bidcheck != None:
            if bidcheck.type == OrderType.LIMIT:
                self.bid_book.remove(bidcheck)
            elif bidcheck.type == OrderType.MARKET:
                self.bid_book.remove(bidcheck)
            elif bidcheck.type == OrderType.IOC:
                self.bid_book.remove(bidcheck)

        elif askcheck != None:
            if askcheck.type == OrderType.LIMIT:
                self.ask_book.remove(askcheck)
            elif askcheck.type == OrderType.MARKET:
                self.ask_book.remove(askcheck)
            elif askcheck.type == OrderType.IOC:
                self.ask_book.remove(askcheck)


# -----------------------------------------------------------

# Each trader can take a separate action chosen from the list below:

# Actions:
# 1 - Place New Order/Order Filled
# 2 - Amend Quantity Of An Existing Order
# 3 - Cancel An Existing Order
# 4 - Return Balance And Position

# request - (Action #, Trader ID, Additional Arguments)

# result - (Action #, Action Return)

# WE ASSUME 'AAPL' IS THE ONLY TRADED STOCK.


class Trader(MyThread):
    def __init__(self, id):
        super().__init__(id)
        self.book_position = 0
        self.balance_track = [1000000]
        # the traders each start with a balance of 1,000,000 and nothing on the books
        # each trader is a thread
        self.ordered = False

    def place_limit_order(self, quantity=None, price=None, side=None):
        # Make sure the limit order given has the parameters necessary to construct the order
        # It's your choice how to implement the orders that do not have enough information

        # The 'order' returned must be of type LimitOrder

        # Make sure you modify the book position after the trade
        # You must return a tuple of the following:
        # (the action type enum, the id of the trader, and the order to be executed)
        trader_to_exchange.appendleft((1, self.id, LimitOrder(id, 'AAPL', quantity, price, side, time.time())))

    def place_market_order(self, quantity=None, side=None):
        # Make sure the market order given has the parameters necessary to construct the order
        # It's your choice how to implement the orders that do not have enough information

        # The 'order' returned must be of type MarketOrder

        # Make sure you modify the book position after the trade
        # You must return a tuple of the following:
        # (the action type enum, the id of the trader, and the order to be executed)
        trader_to_exchange.appendleft((1, self.id, MarketOrder(id, 'AAPL', quantity, side, time.time())))

    def place_ioc_order(self, quantity=None, price=None, side=None):
        # Make sure the ioc order given has the parameters necessary to construct the order
        # It's your choice how to implement the orders that do not have enough information

        # The 'order' returned must be of type IOCOrder

        # Make sure you modify the book position after the trade
        # You must return a tuple of the following:
        # (the action type enum, the id of the trader, and the order to be executed)
        trader_to_exchange.appendleft((1, self.id, IOCOrder(id, 'AAPL', quantity, price, side, time.time())))

    def amend_quantity(self, quantity=None):
        # It's your choice how to implement the 'Amend' action where quantity is not given

        # You must return a tuple of the following:
        # (the action type enum, the id of the trader, and quantity to change the order by)
        trader_to_exchange.appendleft((2, self.id, quantity))

    def cancel_order(self):
        # You must return a tuple of the following:
        # (the action type enum, the id of the trader)
        trader_to_exchange.appendleft((3, self.id))

    def balance_and_position(self):
        # You must return a tuple of the following:
        # (the action type enum, the id of the trader)
        trader_to_exchange.appendleft((4, self.id))

    def process_response(self, response):
        # Implement this function
        # You need to process each order according to the type (by enum) given by the 'response' variable
        if response[0] == 1:
            if response[1].side == OrderSide.BUY:
                self.book_position += response[1].quantity
                self.balance_track.append(self.balance_track[-1] - response[1].quantity * response[1].price)
            elif response[1].side == OrderSide.SELL:
                self.book_position -= response[1].quantity
                self.balance_track.append(self.balance_track[-1] + response[1].quantity * response[1].price)
        elif response[0] == 2:
            pass
        elif response[0] == 3:
            self.ordered = False
        elif response[0] == 4:
            pass
        else:
            raise UndefinedResponse("Undefined Response Received!")
        # if response[0] == 4:
        #    self.balance_track = response[1][1]
        # If the action taken by the trader is ambiguous you need to raise the following error


    def random_action(self):
        # Implement this function
        # According to the status of whether you have a position on the book and the action chosen
        # the trader needs to be able to take a separate action
        if self.book_position == 0:
            myrand = random.randint(1, 4)
            if myrand == 1 and not self.ordered:
                self.place_limit_order(2, random.randint(1, 50), side=OrderSide.BUY)
                self.ordered = True
            elif myrand == 2 and not self.ordered:
                self.place_market_order(2, side=OrderSide.BUY)
            elif myrand == 3 and not self.ordered:
                self.place_ioc_order(2, random.randint(1, 50), side=OrderSide.BUY)
            elif myrand == 4:
                self.balance_and_position()
        else:
            myrand = random.randint(1, 6)
            if myrand == 1 and not self.ordered:
                self.place_limit_order(2, random.randint(1, 50),
                                       OrderSide.BUY if random.randint(1, 2) == 1 else OrderSide.SELL)
                self.ordered = True
            elif myrand == 2 and not self.ordered:
                self.place_market_order(2, OrderSide.BUY if random.randint(1, 2) == 1 else OrderSide.SELL)
            elif myrand == 3 and not self.ordered:
                self.place_ioc_order(2, random.randint(1, 50),
                                     OrderSide.BUY if random.randint(1, 2) == 1 else OrderSide.SELL)
            elif myrand == 4:
                self.amend_quantity(1)
            elif myrand == 5:
                self.balance_and_position()
            elif myrand == 6:
                self.cancel_order()
                self.ordered = False

        # The action taken can be random or deterministic, your choice

    def run_infinite_loop(self):
        # The trader needs to continue to take actions until the book balance falls to 0
        # While the trader can take actions, it chooses from a random_action and uploads the action
        # to the exchange
        # if self.balance_track: return
        if self.balance_track[-1] > 0:
            self.random_action()
            if exchange_to_trader[self.id]:
                response_from_server = exchange_to_trader[self.id].pop()
                self.process_response(response_from_server)
        else:
            return

        # The trader then takes any received responses from the exchange and processes it


class Exchange(MyThread):
    def __init__(self):
        super().__init__()
        self.balance = [1000000 for _ in range(100)]
        self.position = [0 for _ in range(100)]
        self.matching_engine = MatchingEngine()
        # The exchange keeps track of the traders' balances
        # The exchange uses the matching engine you built previously

    def place_new_order(self, order):
        # The exchange must use the matching engine to handle orders given
        success = self.matching_engine.handle_order(order[2])

        results = []
        # The list of results is expected to contain a tuple of the follow form:
        # (Trader id that processed the order, (action type enum, order))
        if success:
            results.append((order[1], (order[0], order[2])))
            for s in success:
                results.append((exchange_to_trader.index(s), (1, s[0])))
                if order[2].side == OrderSide.BUY:
                    self.balance[order[1]] -= s[0].quantity * s[0].price
                    self.balance[exchange_to_trader.index(s)] += s[0].quantity * s[0].price
                    self.position[order[1]] += s[0].quantity
                    self.position[exchange_to_trader.index(s)] -= s[0].quantity
                else:
                    self.balance[order[1]] += s[0].quantity * s[0].price
                    self.balance[exchange_to_trader.index(s)] -= s[0].quantity * s[0].price
                    self.position[order[1]] -= s[0].quantity
                    self.position[exchange_to_trader.index(s)] += s[0].quantity
        # The exchange must update the balance of positions of each trader involved in the trade (if any)

        return results

    def amend_quantity(self, id, quantity):
        # The matching engine must be able to process the 'amend' action based on the given parameters

        # Keep in mind of any exceptions that may be thrown by the matching engine while handling orders
        # The return must be in the form (action type enum, logical based on if order processed)
        return 2, True if self.matching_engine.amend_quantity(id, quantity) else 2, False

    def cancel_order(self, id):
        # The matching engine must be able to process the 'cancel' action based on the given parameters

        # Keep in mind of any exceptions that may be thrown by the matching engine while handling orders
        # The return must be in the form (action type enum, logical based on if order processed)
        return 3, True if self.matching_engine.cancel_order(id) else 3, False

    def balance_and_position(self, id):
        # The matching engine must be able to process the 'balance' action based on the given parameters

        # The return must be in the form (action type enum, (trader balance, trader positions))
        return 4, (self.balance[id], self.position[id])

    def handle_request(self, request):
        # The exchange must be able to process different types of requests based on the action
        # type given using the functions implemented above
        if request[0] == 1:
            return self.place_new_order(request)
        elif request[0] == 2:
            return self.amend_quantity(request[1], request[2])
        elif request[0] == 3:
            return self.cancel_order(request[1])
        elif request[0] == 4:
            return self.balance_and_position(request[1])
            # You must raise the following exception if the action given is ambiguous
        else:
            raise UndefinedTraderAction("Undefined Trader Action!")

    def run_infinite_loop(self):
        # The exchange must continue handling orders as orders are issued by the traders
        # A way to do this is check if there are any orders waiting to be processed in the deque
        # if self.balance: return
        if all(i <= 0 for i in self.balance):
            #self.is_started = False
            return

        while len(trader_to_exchange):
            request = trader_to_exchange.pop()
            return_function = self.handle_request(request)
            if return_function is not None:
                if type(return_function) == list:
                    for r in return_function:
                        exchange_to_trader[request[1]].appendleft(r)
                else:
                    exchange_to_trader[request[1]].appendleft(return_function)

        # If there are, handle the request using the functions built above and using the
        # corresponding trader's deque, return an acknowledgement based on the response


if __name__ == "__main__":

    trader = [Trader(i) for i in range(100)]
    exchange = Exchange()

    exchange.start()
    for t in trader:
        t.start()

    exchange.join()
    for t in trader:
        t.join()

    sum_exch = 0
    for t in MyThread.list_of_threads:
        if t.id == "NoID":
            for b in t.balance:
                sum_exch += b

    print("Total Money Amount for All Traders before Trading Session: " + str(sum_exch))

    for i in range(10000):
        thread_active = False
        for t in MyThread.list_of_threads:
            if t.is_started:
                t.run_infinite_loop()
                thread_active = True
        if not thread_active:
            break

    sum_exch = 0
    for t in MyThread.list_of_threads:
        if t.id == "NoID":
            for b in t.balance:
                sum_exch += b

    print("Total Money Amount for All Traders after Trading Session: ", str(int(sum_exch)))
