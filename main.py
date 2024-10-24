import constants
from model import Model
from tqdm import tqdm


def main() -> None:
  model = Model()
  if constants.LOAD_DATA:
    model.load('model.npy')
  else:
    for _ in tqdm(range(constants.ROUNDS), desc='Rounds'):
      model.simulate_one_round()
    model.save('model.npy')
  model.plot_index()
  model.plot_stock(0)
  model.plot_stock(200)


if __name__ == '__main__':
  main()
