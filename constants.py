import numpy as np


# hyperparameters
LOAD_DATA = False
ROUNDS = 100
SEED = 42


class Agent:
  # Sequence: random agents, expert agents, local imitative agents, market imitative agents, TWAP agents, VWAP agents
  NAMES = [
    'RA',
    # 'EA',
    # 'LIA',
    # 'MIA',
    # 'TWAP',
    # 'VWAP',
  ]
  NUM_TYPE = len(NAMES)
  ID = dict(zip(NAMES, range(NUM_TYPE)))

  NUMS = [2000, 100, 500, 500]
  __BEGIN_INDICES = [0] + list(np.cumsum(NUMS))
  SLICES = []
  for i in range(NUM_TYPE):
    SLICES.append(slice(__BEGIN_INDICES[i], __BEGIN_INDICES[i + 1]))
  TOTAL_NUM = np.sum(NUMS)

  INIT_ASSETS = [2_000_000.0, 10_000_000.0, 2_000_000.0, 2_000_000.0]
  MAX_BUDGET = 10_000.0

  RISK_FREE_INTEREST_RATE = 0
  RISK_FREE_DAILY_RETURN_RATE = .0001

  ALTERNATIVE_NUM = 10
  MAX_PURCHASE_NUM = 5
  EXPECTED_PROFIT = 0

  LOCAL_IMITATIVE_SIGHT = 10


class Stock:
  # Sequence: small, medium, large
  NAMES = [
    'S',
    'M',
    'L',
  ]
  NUM_TYPE = len(NAMES)
  ID = dict(zip(NAMES, range(NUM_TYPE)))

  NUMS = [100, 100, 100]
  __BEGIN_INDICES = [0] + list(np.cumsum(NUMS))
  SLICES = []
  for i in range(NUM_TYPE):
    SLICES.append(slice(__BEGIN_INDICES[i], __BEGIN_INDICES[i + 1]))
  TOTAL_NUM = np.sum(NUMS)

  INIT_PRICES = [10.0, 20.0, 30.0]
  INIT_QUANTITIES = [40_000_000, 80_000_000, 100_000_000]
