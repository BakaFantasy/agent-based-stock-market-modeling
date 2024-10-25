import constants
from model import Model
from tqdm import tqdm


def main() -> None:
  model = Model()
  if constants.LOAD_DATA:
    model.load()
  else:
    for _ in tqdm(range(constants.ROUNDS), desc='Rounds'):
      model.simulate_one_round()
    model.save()
  model.plot_index()
  model.plot_agent(0)


if __name__ == '__main__':
  main()
