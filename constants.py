import numpy as np


# hyperparameters
LOAD_DATA = False
ROUNDS = 10
SEED = 42

# constants
GENERATOR = np.random.Generator(np.random.PCG64(seed=SEED))


class Agent:
  # Sequence: random agents, expert agents, local imitative agents, market imitative agents
  ID = {'RA': 0, 'EA': 1, 'LIA': 2, 'MIA': 3}
  NUM_TYPE = len(ID)
  NUMS = [2000, 100, 500, 500]
  __BEGIN_INDICES = [0] + list(np.cumsum(NUMS))
  SLICES = []
  for i in range(NUM_TYPE):
    SLICES.append(slice(__BEGIN_INDICES[i], __BEGIN_INDICES[i + 1]))
  TOTAL_NUM = np.sum(NUMS)

  INIT_ASSETS = [2_000_000.0, 10_000_000.0, 2_000_000.0, 2_000_000.0]
  MAX_BUDGET = 1_000_000.0

  RISK_FREE_RATE = 0

  ALTERNATIVE_NUM = 10
  MAX_PURCHASE_NUM = 5
  EXPECTED_PROFIT = 0


class Stock:
  # Sequence: small, medium, large
  ID = {'S': 0, 'M': 1, 'L': 2}
  NUM_TYPE = len(ID)
  NUMS = [100, 100, 100]
  __BEGIN_INDICES = [0] + list(np.cumsum(NUMS))
  SLICES = []
  for i in range(NUM_TYPE):
    SLICES.append(slice(__BEGIN_INDICES[i], __BEGIN_INDICES[i + 1]))
  TOTAL_NUM = np.sum(NUMS)

  INIT_PRICES = [10.0, 20.0, 30.0]
  INIT_QUANTITIES = [40_000_000, 80_000_000, 100_000_000]
