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
    for i in range(constants.Agent.NUM_TYPE):
      self.__updaters[i](self.__assets)

  def __order(self) -> (NDArray[np.float64], NDArray[np.float64]):
    # argsort the difference between the expected price and the current price and get the first ALTERNATIVE_NUM stocks
    alternatives = np.argsort(self.__assets[-1, :-1, 1] - self.__assets[:-1, :-1, 1],
                              axis=1)[:, :constants.Agent.ALTERNATIVE_NUM]
    # create purchase order for the stocks in [0, min(MAX_PURCHASE_ORDERS, POSITIVE_COUNT)) if the difference is positive; otherwise create sell order if the agent owns the stock
    purchase_nums = np.array([min(constants.Agent.MAX_PURCHASE_NUM,
                                  np.sum(self.__assets[i, alternatives[i], 1] - self.__assets[-1, alternatives[i], 1] >= constants.Agent.EXPECTED_PROFIT))
                              for i in range(constants.Agent.TOTAL_NUM)], dtype=np.int32)
    purchase_alternatives = np.array([alternatives[i, :purchase_nums[i]]
                                      for i in range(constants.Agent.TOTAL_NUM)], dtype=object)
    # get the stock_ids that the agents owns, which is the stocks with assets[i, stock_id, 0] > 0
    owned_stocks = np.array([np.nonzero(self.__assets[i, :-1, 0] > 0)[0]
                             for i in range(constants.Agent.TOTAL_NUM)], dtype=object)
    # sell_alternatives is the intersection of [alternative[i, purchase_nums[i]:] and owned_stocks[i]
    sell_alternatives = np.array([np.intersect1d(alternatives[i, purchase_nums[i]:], owned_stocks[i])
                                  for i in range(constants.Agent.TOTAL_NUM)], dtype=object)
    # columns of purchase_orders is (agent_id, stock_id, purchase_price), where order_price is current price plus a random number, making the order price in range [current price, current price + (expected price - expected price at purchase_num))
    purchase_orders = np.array([[i, stock_id, self.__assets[-1, stock_id, 1] +
                                 ((self.__assets[i, stock_id, 1] - self.__assets[-1, stock_id, 1]) -
                                  (self.__assets[i, alternatives[i][purchase_nums[i]], 1] -
                                   self.__assets[-1, alternatives[i][purchase_nums[i]], 1])) *
                                 constants.GENERATOR.random()]
                                for i, stock_ids in enumerate(purchase_alternatives)
                                for stock_id in stock_ids])
    sell_orders = np.array([[i, stock_id, self.__assets[-1, stock_id, 1] +
                             ((self.__assets[i, stock_id, 1] - self.__assets[-1, stock_id, 1]) -
                              (self.__assets[i, alternatives[i][purchase_nums[i] - 1], 1] -
                               self.__assets[-1, alternatives[i][purchase_nums[i] - 1], 1])) *
                             constants.GENERATOR.random()]
                            for i, stock_ids in enumerate(sell_alternatives)
                            for stock_id in stock_ids])
    return purchase_orders, sell_orders

  def __negotiate(self, purchase_orders: NDArray[np.float64], sell_orders: NDArray[np.float64]) -> NDArray[np.int32]:
    pass

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
