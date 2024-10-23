import constants
from model import Model


def main() -> None:
  model = Model()
  for _ in range(constants.ROUNDS):
    model.simulate_one_round()
  model.statistics()


if __name__ == '__main__':
  main()
