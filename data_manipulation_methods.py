import pandas as pd
import numpy as np
import gzip
import json
import re
from tqdm import tqdm
from typing import Tuple

# Methods for reading data
# Base code provided code from https://nijianmo.github.io/amazon/index.html

def getDF(path: str, max_lines:int = None,
          V_label='vote'):
  df = {}
  n_problem_lines = 0
  if max_lines is not None:
    max_lines = int(max_lines)
  # open with gzip
  g = gzip.open(path, 'rb')
  pbar = tqdm(enumerate(g), miniters=5000, total=max_lines)
  for i,l in pbar:
    if max_lines is not None and i >= max_lines:
      break
    try:
      d = json.loads(l)
      df[i] = d
    except:
      n_problem_lines += 1
    pbar.set_description(f"Num problem lines: {n_problem_lines}", False)
  # close, dont leave haning
  g.close()
  # create frames
  df = pd.DataFrame.from_dict(df, orient='index')
  # clean data
  df[V_label] = df[V_label].str.replace(',','', regex=True).astype(float)
  df[V_label].fillna(0, inplace=True)
  df[V_label] = df[V_label].astype(int)

  return df

# Methods for injecting synthetic causality

def label_Y(df: pd.DataFrame, min_stars: int = 4,
           new_label: str = 'above3Stars',
           ratings_key = 'overall'):
  """
  Adds a column for Y data (high rating reviews)
  Approx. by having min number of stars
  """
  df[new_label] = df[ratings_key] >= min_stars

def generate_synthetic_text(df: pd.DataFrame, perturb_ratio: float = 0.5,
                            text_label: str = 'reviewText',
                            type_label: str = 'syntheticType',
                            perturb_text_label: str = 'syntheticText',
                            counterfactual_text_label: str = 'cfSyntheticText'):
  """
  Execute different types of perturbation based on type ratios
  """
  # assign Z
  num_entries = df.shape[0]
  df[type_label] = np.random.choice(2, num_entries, p=[perturb_ratio, 1-perturb_ratio])

  # regex
  a_regex = re.compile(r'\ba\b', re.S)
  the_regex = re.compile(r'\bthe\b', re.S)

  # perturb sequence
  perturbed_texts, counterfactual_texts = [], []
  for perturb_type, target_text in zip(df[type_label], df[text_label]):
    if not isinstance(target_text, str):
      perturbed_text = ''
      perturbed_texts.append(perturbed_text)
      counterfactual_texts.append(perturbed_text)
    else:
      # decide Z grouping
      group_0 = perturbed_texts if perturb_type == 0 else counterfactual_texts
      group_1 = counterfactual_texts if perturb_type == 0 else perturbed_texts

      # perturb at specific points
      x_perturbed_text = a_regex.sub(lambda m: m.group().replace('a',"axxxxx",6), target_text)
      x_perturbed_text = the_regex.sub(lambda m: m.group().replace('the',"thexxxxx",8), x_perturbed_text)
      z_perturbed_text = a_regex.sub(lambda m: m.group().replace('a',"azzzzz",6), target_text)
      z_perturbed_text = the_regex.sub(lambda m: m.group().replace('the',"thezzzzz",8), z_perturbed_text)

      # store
      group_0.append(x_perturbed_text)
      group_1.append(z_perturbed_text)

  # add data to df
  df[perturb_text_label] = perturbed_texts
  df[counterfactual_text_label] = counterfactual_texts

def subsample_data(df: pd.DataFrame, gamma: float = 0.3,
                   Y_label: str = 'above3Stars', Z_label: str = 'perturbType'):
  """
  Subsample data to get P(Y=1|Z=1) = P(Y=0|Z=0) = gamma
  """

  def calc_portion(gamma, num_Zz, num_Yy_Zz):
    """
    Calculates proportion of N_Z, and N_Y_Z
    """
    if gamma * num_Zz > num_Yy_Zz:
      # need remove Zz
      delta = np.round(num_Zz - num_Yy_Zz / gamma)
      num_Zz = num_Zz - delta
    else:
      # need remove Yy_Zz (which also means Zz is removed)
      delta = np.round((gamma * num_Zz - num_Yy_Zz)/(gamma - 1))
      num_Zz = num_Zz - delta
      num_Yy_Zz = num_Yy_Zz - delta
    return num_Zz, num_Yy_Zz

  # find best proportion for Y1 Z1
  Z1 = df[Z_label] == 1
  N_Z1 = Z1.sum()
  Y1_Z1 = df[Y_label] & Z1
  N_Y1_Z1 = Y1_Z1.sum()
  Z1_req, Y1_Z1_req = calc_portion(gamma, N_Z1, N_Y1_Z1)

  # find best proportion for Y0 Z0
  Z0 = ~Z1
  N_Z0 = Z0.sum()
  Y0_Z0 = ~df[Y_label] & Z0
  N_Y0_Z0 = Y0_Z0.sum()
  Z0_req, Y0_Z0_req = calc_portion(gamma, N_Z0, N_Y0_Z0)

  # data balancing (P(Y) = 0.5)
  N_Z_req = int(min(Z1_req, Z0_req))
  N_Y_Z_req = int(min(Y1_Z1_req, Y0_Z0_req))
  N_Y_Z_req_inv = N_Z_req - N_Y_Z_req
  # sub sampling
  Y1_Z1_sel = np.random.permutation(df[Y1_Z1].index.to_numpy())[N_Y_Z_req:].tolist()
  Y0_Z1 = ~df[Y_label] & Z1
  Y0_Z1_sel = np.random.permutation(df[Y0_Z1].index.to_numpy())[N_Y_Z_req_inv:].tolist()

  Y0_Z0_sel = np.random.permutation(df[Y0_Z0].index.to_numpy())[N_Y_Z_req:].tolist()
  Y1_Z0 = df[Y_label] & Z0
  Y1_Z0_sel = np.random.permutation(df[Y1_Z0].index.to_numpy())[N_Y_Z_req_inv:].tolist()

  df.drop(Y1_Z1_sel+Y0_Z1_sel+Y0_Z0_sel+Y1_Z0_sel, inplace=True)

def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8, gamma: float = 0.3,
                  epsilon = 0.03, max_iter: int = 100,
                  Y_label: str = 'above3Stars', Z_label: str = 'perturbType'):
  """
  Split dataset to training and testing
  Resample until P(Y=1|Z=1) = P(Y=0|Z=0) = gamma is within a threshold
  """
  N = df.shape[0]
  N_test = int(np.round(train_ratio * N))
  success = False
  for _ in range(max_iter):
    # try one split
    rand_seq = np.random.permutation(N)
    train_df = df.iloc[rand_seq[:N_test]]
    test_df = df.iloc[rand_seq[N_test:]]

    # check within range
    Z1_train = train_df[Z_label] == 1
    Z0_train = ~Z1_train
    P_V_Z1_train = (Z1_train & train_df[Y_label]).sum() / Z1_train.sum()
    P_V_Z0_train = (Z0_train & (~train_df[Y_label])).sum() / Z0_train.sum()
    Z1_test = test_df[Z_label] == 1
    Z0_test = ~Z1_test
    P_V_Z1_test = (Z1_test & test_df[Y_label]).sum() / Z1_test.sum()
    P_V_Z0_test = (Z0_test & (~test_df[Y_label])).sum() / Z0_test.sum()
    if (np.abs(P_V_Z1_train - gamma) < epsilon) and (np.abs(P_V_Z0_train - gamma) < epsilon) and \
      (np.abs(P_V_Z1_test - gamma) < epsilon) and (np.abs(P_V_Z0_test - gamma) < epsilon):
      success = True
      break

  if not success:
    raise Exception("Max iter reached without hitting gamma condition")

  return train_df, test_df


# Methods for natural causal selection

def drop_data(df: pd.DataFrame, gamma: float = 0.3,
              Z_label:str ='above3Stars', V_label: str='vote',
              max_iter:int = 100):
  """
  Drop reviews with no votes untill conditions are met
  1. P(V>0| Z=1) > gamma
  2. P(V>0| Z=0) > 1- gamma
  """

  success = False

  # perform up to max_iter of dropping (more than 1 drop can happen each iter)
  for _ in range(max_iter):
    # calculate gamma condition
    N_Z1 = df[Z_label].sum()
    N_V_Z1 = (df[Z_label] & (df[V_label]>0)).sum()
    t1 = (df[df[Z_label]][V_label] > 0).sum()
    P_V_Z1 = N_V_Z1 / N_Z1

    N_Z0 = (~df[Z_label]).sum()
    N_V_Z0 = (~df[Z_label] & (df[V_label]>0)).sum()
    P_V_Z0 = N_V_Z0 / N_Z0

    # break if gamma condition met
    if (P_V_Z1 > gamma and P_V_Z0 > (1 - gamma)):
      success = True
      break

    # calculate min entities to drop
    min_drop_V_Z1 = np.ceil(N_Z1 - N_V_Z1 / gamma)
    min_drop_V_Z0 = np.ceil(N_Z0 - N_V_Z0 / (1-gamma))
    min_drop = max(min_drop_V_Z1, min_drop_V_Z0)

    # perform drop
    droppable = df[V_label]==0
    N_droppable = droppable.sum()
    if N_droppable < min_drop: # check if enough to drop
      raise Exception("No more entities to drop to hit gamma condition")
    rand_seq = np.random.choice(2,N_droppable,
                                p=[(N_droppable-min_drop)/N_droppable, min_drop/N_droppable]).astype(bool)
    df.drop(df[droppable].index[rand_seq], inplace=True)

  # max iter hit without reaching gamma condition
  if not success:
    raise Exception("Max iter reached without hitting gamma condition")


def find_smallest_Tz(df: pd.DataFrame, gamma: float = 0.3,
                     Z_label:str ='above3Stars', V_label: str='vote') -> Tuple[int, int]:
  """
  Find smallest T1 and T0 such that:
  1. P(V>T1|Z=1) < gamma
  2. P(V>T0|Z=0) < 1 - gamma
  """

  def scan_Tz(filtered_df: pd.DataFrame, threshold: float,
              V_label: str, T_max: int):
    N = filtered_df.shape[0]
    for T in range(T_max):
      if float((filtered_df[V_label] > T).sum()) / N < threshold:
        return T

    assert False, "Max iteration reached without hitting condition!!"

  T_max = df[V_label].max()
  T_1 = scan_Tz(df[df[Z_label]], gamma, V_label, T_max)
  T_0 = scan_Tz(df[~df[Z_label]], 1-gamma, V_label, T_max)

  return T_1, T_0

def assign_natural_Y(df: pd.DataFrame, T_1: int, T_0: int,
                     gamma: float = 0.3,
                     Z_label:str ='above3Stars', V_label: str='vote',
                     Y_label:str ='aboveVThreshold'):

  def random_flip(df: pd.DataFrame, filter: pd.Series,
                  T_k: int, threshold: float,
                  V_label: str, Y_label:str):
    N = filter.sum()
    min_count_req = int(np.ceil(threshold * N))
    current_count = df[filter][Y_label].sum()

    if current_count <= min_count_req:
      available_flips = df[filter][V_label] == T_k
      required_flips = min_count_req - current_count

      assert required_flips <= available_flips.sum(), "Not enough Tk+1 to hit threshold!"

      to_flip = df[filter][available_flips].sample(required_flips).index
      df.loc[to_flip, Y_label] = True

  # init to Y=1 if V>T_1|Z=1 or V>T_0|Z=0
  df = df.assign(**{Y_label: (df[Z_label] & (df[V_label] > T_1)) | (~df[Z_label] & (df[V_label] > T_0))})

  # randomly flip Y=0 -> Y=1 for V=T_1+1,Z=1 until P(Y=1|Z=1)>gamma
  random_flip(df, df[Z_label], T_1, gamma, V_label, Y_label)

  # randomly flip Y=0 -> Y=1 for V=T_0+1,Z=0 until P(Y=1|Z=0)>1-gamma
  random_flip(df, ~df[Z_label], T_0, 1 - gamma, V_label, Y_label)

  return df


## Methods for saving dataset

def save_df(df: pd.DataFrame, file_name: str, text_keys= ['reviewText', 'syntheticText', 'cfSyntheticText']):
  """
  Saves dataframe into a npz file with keys and data
  - text_keys : key of entries that needs to be converted from obj to string
  """
  # convert to str type
  [df[text_key].astype(str) for text_key in text_keys]
  # save data
  np.savez(file_name, **{key: df[key] for key in df.keys()})