import numpy as np


# hyperparameters
ROUNDS = 100
SEED = 42
LOAD_DATA = False
ASSETS_PATH = 'data/assets.npy'
TRANSACTIONS_PATH = 'data/transactions.npy'


class Agent:
  class Random:
    NUM = 2000
    SLICE = None
    INIT_ASSETS = 2_000_000.0

  class Expert:
    NUM = 100
    SLICE = None
    INIT_ASSETS = 10_000_000.0

  class LocalImitative:
    NUM = 500
    SLICE = None
    INIT_ASSETS = 2_000_000.0
    SIGHT = 10

  class MarketImitative:
    NUM = 500
    SLICE = None
    INIT_ASSETS = 2_000_000.0

  class TimeWeighted:
    NUM = 500
    SLICE = None
    INIT_ASSETS = 2_000_000.0
    SIGHT = 10

  class VolumeWeighted:
    NUM = 500
    SLICE = None
    INIT_ASSETS = 2_000_000.0
    SIGHT = 10

  SELECTIONS = [
    Random,
    Expert,
    # LocalImitative,
    # MarketImitative,
    # TimeWeighted,
    # VolumeWeighted,
  ]

  NUM_SELECTION = len(SELECTIONS)
  __BEGIN_INDICES = [0] + list(np.cumsum([selection.NUM for selection in SELECTIONS]))
  for i in range(NUM_SELECTION):
    SELECTIONS[i].SLICE = slice(__BEGIN_INDICES[i], __BEGIN_INDICES[i + 1])
  TOTAL_NUM = np.sum([selection.NUM for selection in SELECTIONS])

  MAX_BUDGET = 100_000.0

  RISK_FREE_INTEREST_RATE = 0
  RISK_FREE_DAILY_RETURN_RATE = .0001

  ALTERNATIVE_NUM = 10
  MAX_PURCHASE_NUM = 5
  EXPECTED_PROFIT = 0


class Stock:
  class Small:
    NUM = 100
    SLICE = None
    INIT_PRICE = 10.0
    INIT_QUANTITY = 40_000_000

  class Medium:
    NUM = 100
    SLICE = None
    INIT_PRICE = 20.0
    INIT_QUANTITY = 80_000_000

  class Large:
    NUM = 100
    SLICE = None
    INIT_PRICE = 30.0
    INIT_QUANTITY = 100_000_000

  SELECTIONS = [
    Small,
    Medium,
    Large,
  ]

  NUM_SELECTION = len(SELECTIONS)
  __BEGIN_INDICES = [0] + list(np.cumsum([selection.NUM for selection in SELECTIONS]))
  for i in range(NUM_SELECTION):
    SELECTIONS[i].SLICE = slice(__BEGIN_INDICES[i], __BEGIN_INDICES[i + 1])
  TOTAL_NUM = np.sum([selection.NUM for selection in SELECTIONS])
