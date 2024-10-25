import os
import constants
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray


# TODO: substitute iterations with numpy ufuncs as much as possible
class Model:
  def __init__(self):
    self._assets = np.zeros((constants.Agent.TOTAL_NUM + 1, constants.Stock.TOTAL_NUM + 1, 2),
                            dtype=np.float64)
    # initialize agents and the market
    for i in range(constants.Agent.NUM_SELECTION):
      self._assets[constants.Agent.SELECTIONS[i].SLICE, -1, 0] = constants.Agent.SELECTIONS[i].INIT_ASSETS
      self._assets[constants.Agent.SELECTIONS[i].SLICE, -1, 1] = 1 + constants.Agent.RISK_FREE_INTEREST_RATE
    for i in range(constants.Stock.NUM_SELECTION):
      self._assets[:, constants.Stock.SELECTIONS[i].SLICE, 1] = constants.Stock.SELECTIONS[i].INIT_PRICE
      self._assets[-1, constants.Stock.SELECTIONS[i].SLICE, 0] = constants.Stock.SELECTIONS[i].INIT_QUANTITY
    self._assets[-1, -1] = [0, 1]

    # set random generator
    self._generator = np.random.Generator(np.random.PCG64(seed=constants.SEED))

    # set updaters
    self._updaters = [self._update_random_agents, self._update_expert_agents,
                      self._update_local_imitative_agents, self._update_market_imitative_agents,
                      self._update_time_weighted_agents, self._update_volume_weighted_agents]

    # set history
    self._history_assets = []
    self._history_transactions = []

  def _update_random_agents(self) -> None:
    individual_expectations = self._assets[constants.Agent.Random.SLICE, :-1, 1] * (1 + 2 * (
            self._generator.random(size=(constants.Agent.Random.NUM, constants.Stock.TOTAL_NUM)) - .5) * .1)
    expert_mean = np.mean(self._assets[constants.Agent.Expert.SLICE, :-1, 1])
    self._assets[constants.Agent.Random.SLICE, :-1, 1] = (individual_expectations +
                                                          (expert_mean - individual_expectations) * .2)

  def _update_expert_agents(self) -> None:
    self._assets[constants.Agent.Expert.SLICE, :-1, 1] = (
      self._generator.normal(loc=0, scale=.01, size=(constants.Agent.Expert.NUM, constants.Stock.TOTAL_NUM)) +
      constants.Agent.RISK_FREE_DAILY_RETURN_RATE + 1) * self._assets[-1, :-1, 1]

  def _update_local_imitative_agents(self) -> None:
    # retrospecting the last sight rounds, the new expectation is
    # random(sum(purchase price) / sum(purchase quantity), sum(sell price) / sum(sell quantity)), where
    # the sums are over the sight rounds
    sight = min(constants.Agent.LocalImitative.SIGHT, len(self._history_transactions))
    if sight == 0:
      return
    transaction_sums = np.sum(self._history_transactions[-sight:], axis=0)
    for stock_id in range(constants.Stock.TOTAL_NUM):
      if transaction_sums[stock_id, 0] == 0 or transaction_sums[stock_id, 1] == 0:
        continue
      self._assets[constants.Agent.LocalImitative.SLICE, stock_id, 1] = (transaction_sums[stock_id, 2] /
                                                                         transaction_sums[stock_id, 0] +
                                                                         self._generator.random() *
                                                                         (transaction_sums[stock_id, 3] /
                                                                          transaction_sums[stock_id, 1] -
                                                                          transaction_sums[stock_id, 2] /
                                                                          transaction_sums[stock_id, 0]))

  def _update_market_imitative_agents(self) -> None:
    #
    pass

  def _update_time_weighted_agents(self) -> None:
    # retrospecting the last sight rounds, the new expectation is
    # sum(closing price) / sight, where the sums are over the sight rounds
    sight = min(constants.Agent.TimeWeighted.SIGHT, len(self._history_transactions))
    if sight == 0:
      return
    # calculate the average price of the stocks in the last sight rounds
    self._assets[constants.Agent.TimeWeighted.SLICE, :-1, 1] = np.average(self._history_assets[-sight:],
                                                                          axis=0)[-1, :-1, 1]

  def _update_volume_weighted_agents(self) -> None:
    # retrospecting the last sight rounds, the new expectation is
    # sum(closing price * quantity) / sum(quantity), where the sums are over the sight rounds
    sight = min(constants.Agent.VolumeWeighted.SIGHT, len(self._history_transactions))
    if sight == 0:
      return
    transaction_sums = np.sum(self._history_transactions[-sight:], axis=0)
    for stock_id in range(constants.Stock.TOTAL_NUM):
      if transaction_sums[stock_id, 0] == 0 and transaction_sums[stock_id, 1] == 0:
        continue
      self._assets[constants.Agent.VolumeWeighted.SLICE, stock_id, 1] = ((transaction_sums[stock_id, 2] +
                                                                          transaction_sums[stock_id, 3]) /
                                                                         (transaction_sums[stock_id, 0] +
                                                                          transaction_sums[stock_id, 1]))

  def _update(self) -> None:
    """Updates expected prices of all types of agents."""
    for i in range(constants.Agent.NUM_SELECTION):
      self._updaters[i]()

  def _order(self) -> (NDArray[np.float64], NDArray[np.float64]):
    """
    Constructs purchase orders and sell orders for each agent.

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
    alternatives = np.argsort(self._assets[-1, :-1, 1] - self._assets[:-1, :-1, 1],
                              axis=1)[:, :constants.Agent.ALTERNATIVE_NUM]

    purchase_nums = np.array([min(constants.Agent.MAX_PURCHASE_NUM,
                                  np.sum(self._assets[i, alternatives[i], 1] - self._assets[-1, alternatives[i], 1] >=
                                         constants.Agent.EXPECTED_PROFIT))
                              for i in range(constants.Agent.TOTAL_NUM)], dtype=np.int32)

    purchase_alternatives = np.array([np.array(alternatives[i, :purchase_nums[i]], dtype=np.int32)
                                      for i in range(constants.Agent.TOTAL_NUM)], dtype=object)

    owned_stocks = np.array([np.nonzero(self._assets[i, :-1, 0] > 0)[0]
                             for i in range(constants.Agent.TOTAL_NUM + 1)], dtype=object)

    sell_alternatives = np.array([np.setdiff1d(owned_stocks[i], purchase_alternatives[i])
                                  for i in range(constants.Agent.TOTAL_NUM)], dtype=object)

    budgets = np.array([min(self._assets[i, -1, 0], constants.Agent.MAX_BUDGET)
                        for i in range(constants.Agent.TOTAL_NUM)], dtype=np.float64)
    profits = np.array([self._assets[i, np.array(purchase_alternatives[i], dtype=np.int32), 1] -
                        self._assets[-1, np.array(purchase_alternatives[i], dtype=np.int32), 1]
                        for i in range(constants.Agent.TOTAL_NUM)], dtype=object)
    quantities = []
    for i, stock_ids in enumerate(purchase_alternatives):
      # if the sum of profits is 0, the budget is divided equally among the stocks
      # otherwise, the budget is divided proportionally to sigmoid(profits)
      sigmoid_profits = 1 / (1 + np.exp(-profits[i].astype(np.float64)))
      quantities.append(dict(zip(stock_ids, np.floor(budgets[i] * sigmoid_profits / np.sum(sigmoid_profits) /
                                                     self._assets[i, np.array(stock_ids).astype(np.int32), 1]))))
    quantities = np.array(quantities, dtype=object)
    assert np.all([np.sum([quantities[i][stock_id] * self._assets[i, stock_id, 1]
                           for stock_id in purchase_alternatives[i]]) <= budgets[i]
                   for i in range(constants.Agent.TOTAL_NUM)])

    purchase_orders = np.array([(i, stock_id, quantities[i][stock_id],
                                 self._assets[-1, stock_id, 1] +
                                 ((self._assets[i, stock_id, 1] -
                                   self._assets[-1, stock_id, 1]) -
                                  (self._assets[i, alternatives[i][purchase_nums[i]], 1] -
                                   self._assets[-1, alternatives[i][purchase_nums[i]], 1])) *
                                 self._generator.random())
                                for i, stock_ids in enumerate(purchase_alternatives)
                                for stock_id in stock_ids
                                if np.nonzero(quantities[i][stock_id]) and not np.isnan(quantities[i][stock_id])],
                               dtype=np.float64)

    sell_orders = np.array([[i, stock_id, self._assets[i, stock_id, 0],
                             self._assets[-1, stock_id, 1] +
                             ((self._assets[i, stock_id, 1] -
                               self._assets[-1, stock_id, 1]) -
                              (self._assets[i, alternatives[i][purchase_nums[i] - 1], 1] -
                               self._assets[-1, alternatives[i][purchase_nums[i] - 1], 1])) *
                             self._generator.random()]
                            for i, stock_ids in enumerate(sell_alternatives)
                            for stock_id in stock_ids], dtype=np.float64)

    if sell_orders.size == 0:
      sell_orders = np.empty((0, 4), dtype=np.float64)
    market_orders = np.array([[constants.Agent.TOTAL_NUM, stock_id, self._assets[-1, stock_id, 0],
                               self._assets[-1, stock_id, 1]]
                              for stock_id in owned_stocks[-1]])
    sell_orders = np.append(sell_orders, market_orders, axis=0)

    return purchase_orders, sell_orders

  def _negotiate(self, purchase_orders: NDArray[np.float64], sell_orders: NDArray[np.float64]) -> NDArray[np.int32]:
    """
    Matches purchase orders and sell orders to construct transactions.

    The process of constructing transactions is divided into 3 steps:

    1. sort the purchase_orders and sell_orders by the order price in descending order.

    2. group the two arrays of orders by stock numbers.

    3. match the purchase_orders and sell_orders in loop with dual pointers, recording quantities of stocks which are
      successfully transacted for each agent.

    :param purchase_orders: Tuples of (agent_id, stock_id, quantity, purchase_price).
    :param sell_orders: Tuple of (agent_id, stock_id, quantity, sell_price).
    :return: quantities of transactions for each agent.
    """
    purchase_orders = purchase_orders[np.argsort(purchase_orders[:, 3])]
    sell_orders = sell_orders[np.argsort(sell_orders[:, 3])]

    purchase_orders = np.array([purchase_orders[purchase_orders[:, 1] == i]
                                for i in range(constants.Stock.TOTAL_NUM)], dtype=object)
    sell_orders = np.array([sell_orders[sell_orders[:, 1] == i]
                            for i in range(constants.Stock.TOTAL_NUM)], dtype=object)

    transactions = np.zeros((constants.Agent.TOTAL_NUM + 1, constants.Stock.TOTAL_NUM), dtype=np.int32)
    for stock_id in range(constants.Stock.TOTAL_NUM):
      purchase_pointer = purchase_orders[stock_id].shape[0] - 1
      sell_pointer = 0
      closing_price = self._assets[-1, stock_id, 1]
      while (purchase_pointer in range(purchase_orders[stock_id].shape[0]) and
             sell_pointer in range(sell_orders[stock_id].shape[0]) and
             purchase_orders[stock_id][purchase_pointer, 3] >= sell_orders[stock_id][sell_pointer, 3]):
        if (abs(purchase_orders[stock_id][purchase_pointer, 3] - self._assets[-1, stock_id, 1]) <
                abs(sell_orders[stock_id][sell_pointer, 3] - self._assets[-1, stock_id, 1])):
          closing_price = purchase_orders[stock_id][purchase_pointer, 3]
        else:
          closing_price = sell_orders[stock_id][sell_pointer, 3]
        quantity = min(purchase_orders[stock_id][purchase_pointer, 2], sell_orders[stock_id][sell_pointer, 2])
        transactions[int(purchase_orders[stock_id][purchase_pointer, 0]), stock_id] = quantity
        transactions[int(sell_orders[stock_id][sell_pointer, 0]), stock_id] = -quantity
        purchase_orders[stock_id][purchase_pointer, 2] -= quantity
        sell_orders[stock_id][sell_pointer, 2] -= quantity
        if purchase_orders[stock_id][purchase_pointer, 2] == 0:
          purchase_pointer -= 1
        if sell_orders[stock_id][sell_pointer, 2] == 0:
          sell_pointer += 1
      self._assets[-1, stock_id, 1] = closing_price

    return np.array(transactions, dtype=np.float64)

  def _record_transactions(self, transactions: np.array) -> None:
    """
    Records the transactions, aka sum(purchase_quantity), sum(sell_quantity)
    sum(price * purchase_quantity) and sum(price * sell_quantity) for each stock.

    :param transactions: quantities of transactions for each agent.
    """
    purchase_transactions = np.maximum(transactions, 0)
    sell_transactions = np.maximum(-transactions, 0)
    purchase_quantities = np.sum(purchase_transactions, axis=0)
    sell_quantities = np.sum(sell_transactions, axis=0)
    purchase_prices = np.sum(self._assets[:, :-1, 1] * purchase_transactions, axis=0)
    sell_prices = np.sum(self._assets[:, :-1, 1] * sell_transactions, axis=0)
    self._history_transactions.append(np.array([purchase_quantities, sell_quantities,
                                                purchase_prices, sell_prices], dtype=np.float64).T)

  def _transact(self, transactions: np.array) -> None:
    """
    Updates the assets of agents and the market according to the transactions.

    :param transactions: quantities of transactions for each agent.
    """
    self._assets[:, :-1, 0] += transactions
    self._assets[:, -1, 0] -= np.sum(transactions * self._assets[:, :-1, 1], axis=1)
    assert np.all(self._assets >= 0)

  def _record_assets(self) -> None:
    """Records the current state of the market."""
    self._history_assets.append(self._assets.copy())

  def simulate_one_round(self) -> None:
    self._update()
    purchase_orders, sell_orders = self._order()
    transactions = self._negotiate(purchase_orders, sell_orders)
    self._record_transactions(transactions)
    self._transact(transactions)
    self._record_assets()

  def plot_index(self) -> None:
    # TODO: more concise index calculation
    # plot the index of the market, i.e. the changes of _assets[-1, -1, 0]
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(len(self._history_assets)), y=[history[-1, -1, 0] for history in self._history_assets])
    plt.xlabel('Round')
    plt.ylabel('Index')
    plt.title('Index of the Market')
    plt.show()

  def plot_stock(self, stock_id: int) -> None:
    if stock_id not in range(constants.Stock.TOTAL_NUM):
      raise ValueError('stock_id must be in [0, constants.Stock.TOTAL_NUM).')
    # plot the price of the stock with stock_id
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(len(self._history_assets)), y=[history[-1, stock_id, 1] for history in self._history_assets])
    plt.xlabel('Round')
    plt.ylabel('Price')
    plt.title(f'Price of Stock {stock_id}')
    plt.show()

  def plot_agent(self, agent_id: int) -> None:
    if agent_id not in range(constants.Agent.TOTAL_NUM):
      raise ValueError('agent_id must be in [0, constants.Agent.TOTAL_NUM).')
    # plot the assets of the agent with agent_id
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(len(self._history_assets)), y=[history[agent_id, -1, 0] for history in self._history_assets])
    plt.xlabel('Round')
    plt.ylabel('Assets')
    plt.title(f'Assets of Agent {agent_id}')
    plt.show()

  def load(self) -> None:
    if (not os.path.exists(constants.ASSETS_PATH) or
            not os.path.exists(constants.TRANSACTIONS_PATH)):
      raise FileNotFoundError(f'{constants.ASSETS_PATH} not exists.')
    self._history_assets = np.load(constants.ASSETS_PATH)
    self._assets = self._history_assets[-1]
    self._history_transactions = np.load(constants.TRANSACTIONS_PATH)

  def save(self) -> None:
    # check if the directory exists
    def _save_helper(path: str, data: np.array) -> None:
      if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(constants.ASSETS_PATH))
      if os.path.exists(path):
        os.rename(path, '%s.bak' % path)
      np.save(path, data)
    _save_helper(constants.ASSETS_PATH, self._history_assets)
    _save_helper(constants.TRANSACTIONS_PATH, self._history_transactions)
