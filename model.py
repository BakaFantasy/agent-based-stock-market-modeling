import numpy as np
from numpy.typing import NDArray

import constants


def update_normal_agents(assets: NDArray[np.float64]) -> None:
  # individual_expectations is the (2000, 300) matrix, each of which is multiplied by the random number 1 + 2 * (constants.GENERATOR.random(size=constants.Agent.NUMS[0]) - .5) * .1
  individual_expectations = assets[constants.Agent.SLICES[0], :-1, 1] * np.array([
    1 + 2 * (constants.GENERATOR.random(size=constants.Stock.TOTAL_NUM) - .5) * .1
    for _ in range(constants.Agent.NUMS[0])])
  expert_mean = np.mean(assets[constants.Agent.SLICES[1], :-1, 1])
  assets[constants.Agent.SLICES[0], :-1, 1] = individual_expectations + (expert_mean - individual_expectations) * .2


def update_expert_agents(assets: NDArray[np.float64]) -> None:
  assets[constants.Agent.SLICES[1], :-1, 1] = np.array([constants.GENERATOR.random(
    size=constants.Stock.TOTAL_NUM) + constants.Agent.RISK_FREE_RATE + 1
                                                        for _ in range(constants.Agent.NUMS[1])]) * assets[-1, :-1, 1]


def update_local_imitative_agents(assets: NDArray[np.float64]) -> None:
  pass


def update_market_imitative_agents(assets: NDArray[np.float64]) -> None:
  pass


class Model:
  def __init__(self):
    self.__assets = np.zeros((constants.Agent.TOTAL_NUM + 1, constants.Stock.TOTAL_NUM + 1, 2),
                             dtype=np.float64)
    # initialize agents and the market
    for i in range(constants.Agent.NUM_TYPE):
      self.__assets[constants.Agent.SLICES[i], -1, 0] = constants.Agent.INIT_ASSETS[i]
      self.__assets[constants.Agent.SLICES[i], -1, 1] = 1 + constants.Agent.RISK_FREE_RATE
    for i in range(constants.Stock.NUM_TYPE):
      self.__assets[:, constants.Stock.SLICES[i], 1] = constants.Stock.INIT_PRICES[i]
      self.__assets[-1, constants.Stock.SLICES[i], 0] = constants.Stock.INIT_QUANTITIES[i]
    self.__assets[-1, -1] = [0, 1]

    # set updaters
    self.__updaters = [update_normal_agents, update_expert_agents,
                       update_local_imitative_agents, update_market_imitative_agents]

  def __update(self) -> None:
    """
    Update expected prices of all types of agents.
    """
    for i in range(constants.Agent.NUM_TYPE):
      self.__updaters[i](self.__assets)

  def __order(self) -> (NDArray[np.float64], NDArray[np.float64]):
    """
    Construct purchase orders and sell orders for each agent.

    The process of constructing orders is divided into 9 steps:

    1. generate alternatives for each agent, which are the stocks that have the highest expected profit.

    2. determine the number of stocks that each agent wants to purchase.

    3. pick the stocks that each agent wants to purchase from the alternatives.

    4. filter out the stocks that each agent owns.

    5. pick the stocks that each agent wants to sell from the intersection of the alternatives and the owned stocks.

    6. determine quantities of each stock that each agent wants to purchase proportional to the expected profits.

    7. construct purchase orders, which are the tuples of (agent_id, stock_id, quantity, purchase_price).

    8. construct sell orders, which are the tuples of (agent_id, stock_id, quantity, sell_price).

    9. append the market orders to sell orders, where the market sells all the stocks it owns with the current price.

    :return: A tuple of purchase orders and sell orders.
    """
    alternatives = np.argsort(self.__assets[-1, :-1, 1] - self.__assets[:-1, :-1, 1],
                              axis=1)[:, :constants.Agent.ALTERNATIVE_NUM]

    purchase_nums = np.array([min(constants.Agent.MAX_PURCHASE_NUM,
                                  np.sum(self.__assets[i, alternatives[i], 1] - self.__assets[-1, alternatives[i], 1] >= constants.Agent.EXPECTED_PROFIT))
                              for i in range(constants.Agent.TOTAL_NUM)], dtype=np.int32)

    purchase_alternatives = np.array([alternatives[i, :purchase_nums[i]]
                                      for i in range(constants.Agent.TOTAL_NUM)], dtype=object)

    owned_stocks = np.array([np.nonzero(self.__assets[i, :-1, 0] > 0)[0]
                             for i in range(constants.Agent.TOTAL_NUM + 1)], dtype=object)

    sell_alternatives = np.array([np.intersect1d(alternatives[i, purchase_nums[i]:], owned_stocks[i])
                                  for i in range(constants.Agent.TOTAL_NUM)], dtype=object)

    budgets = np.array([min(self.__assets[i, -1, 0], constants.Agent.MAX_BUDGET)
                        for i in range(constants.Agent.TOTAL_NUM)], dtype=np.float64)
    profits = np.array([self.__assets[i, np.array(purchase_alternatives[i], dtype=np.int32), 1] - self.__assets[-1, np.array(purchase_alternatives[i], dtype=np.int32), 1]
                        for i in range(constants.Agent.TOTAL_NUM)], dtype=object)
    # TODO: eliminate division by zero
    quantities = np.array([dict(zip(stock_ids, budgets[i] / np.sum(profits[i]) * profits[i]))
                     for i, stock_ids in enumerate(purchase_alternatives)], dtype=object)

    purchase_orders = np.array([[i, stock_id, np.floor(quantities[i][stock_id]),
                                self.__assets[-1, stock_id, 1] +
                                 ((self.__assets[i, stock_id, 1] - self.__assets[-1, stock_id, 1]) -
                                  (self.__assets[i, alternatives[i][purchase_nums[i]], 1] -
                                   self.__assets[-1, alternatives[i][purchase_nums[i]], 1])) *
                                 constants.GENERATOR.random()]
                                for i, stock_ids in enumerate(purchase_alternatives)
                                for stock_id in stock_ids
                                if np.nonzero(quantities[i][stock_id]) and not np.isnan(quantities[i][stock_id])],
                               dtype=np.float64)

    sell_orders = np.array([[i, stock_id, self.__assets[i, stock_id, 0],
                             self.__assets[-1, stock_id, 1] +
                             ((self.__assets[i, stock_id, 1] - self.__assets[-1, stock_id, 1]) -
                              (self.__assets[i, alternatives[i][purchase_nums[i] - 1], 1] -
                               self.__assets[-1, alternatives[i][purchase_nums[i] - 1], 1])) *
                             constants.GENERATOR.random()]
                            for i, stock_ids in enumerate(sell_alternatives)
                            for stock_id in stock_ids], dtype=np.float64)

    if sell_orders.size == 0:
      sell_orders = np.empty((0, 4), dtype=np.float64)
    market_orders = np.array([[constants.Agent.TOTAL_NUM, stock_id, self.__assets[-1, stock_id, 0],
                               self.__assets[-1, stock_id, 1]]
                              for stock_id in owned_stocks[-1]])
    sell_orders = np.append(sell_orders, market_orders, axis=0)

    return purchase_orders, sell_orders

  def __negotiate(self, purchase_orders: NDArray[np.float64], sell_orders: NDArray[np.float64]) -> NDArray[np.int32]:
    """
    Match purchase orders and sell orders to construct transactions.

    The process of constructing transactions is divided into 3 steps:

    1. sort the purchase_orders and sell_orders by the order price in descending order.

    2. group the two arrays of orders by stock numbers.

    3. match the purchase_orders and sell_orders in loop with dual pointers, constructing transactions
      which are tuples of (purchaser_id, seller_id, stock_id, quantity, price).

    :param purchase_orders: Tuples of (agent_id, stock_id, quantity, purchase_price).
    :param sell_orders: Tuple of (agent_id, stock_id, quantity, sell_price).
    :return: Tuples of (purchaser_id, seller_id, stock_id, quantity, price).
    """
    purchase_orders = purchase_orders[np.argsort(purchase_orders[:, 3])[::-1]]
    sell_orders = sell_orders[np.argsort(sell_orders[:, 3])[::-1]]

    purchase_orders = np.array([purchase_orders[purchase_orders[:, 1] == i]
                                for i in range(constants.Stock.TOTAL_NUM)], dtype=object)
    sell_orders = np.array([sell_orders[sell_orders[:, 1] == i]
                            for i in range(constants.Stock.TOTAL_NUM)], dtype=object)

    transactions = []
    for stock_id in range(constants.Stock.TOTAL_NUM):
      purchase_pointer = 0
      sell_pointer = sell_orders[stock_id].shape[0] - 1
      while (purchase_pointer in range(purchase_orders[stock_id].shape[0]) and
             sell_pointer in range(sell_orders[stock_id].shape[0]) and
             purchase_orders[stock_id][purchase_pointer, 3] >= sell_orders[stock_id][sell_pointer, 3]):
        quantity = min(purchase_orders[stock_id][purchase_pointer, 2], sell_orders[stock_id][sell_pointer, 2])
        # TODO: turn a row into tuple with dtype
        transactions.append([purchase_orders[stock_id][purchase_pointer, 0],
                             sell_orders[stock_id][sell_pointer, 0],
                             stock_id, quantity, sell_orders[stock_id][sell_pointer, 3]])
        purchase_orders[stock_id][purchase_pointer, 2] -= quantity
        sell_orders[stock_id][sell_pointer, 2] -= quantity
        if purchase_orders[stock_id][purchase_pointer, 2] == 0:
          purchase_pointer += 1
        if sell_orders[stock_id][sell_pointer, 2] == 0:
          sell_pointer -= 1
    return np.array(transactions, dtype=np.float64)

  def __transact(self, transactions: np.array) -> None:
    pass

  def simulate_one_round(self) -> None:
    self.__update()
    purchase_orders, sell_orders = self.__order()
    transactions = self.__negotiate(purchase_orders, sell_orders)
    self.__transact(transactions)

  def statistics(self) -> None:
    pass

  def load(self, path: str) -> None:
    self.__assets = np.load(path)

  def save(self, path: str) -> None:
    np.save(path, self.__assets)
