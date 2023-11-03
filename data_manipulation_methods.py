import pandas as pd
import numpy as np
import gzip
import json
import re

# Methods for reading data
# Base code provided code from https://nijianmo.github.io/amazon/index.html

def getDF(path: str, max_lines:int = None,
          V_label='vote'):
  df = {}
  # open with gzip
  g = gzip.open(path, 'rb')
  for i,l in enumerate(g):
    d = json.loads(l)
    df[i] = d
    if max_lines is not None and i > max_lines:
      break
  # close, dont leave haning
  g.close()
  # create frames
  df = pd.DataFrame.from_dict(df, orient='index')
  # clean data
  df[V_label].fillna(0, inplace=True)
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

def synthetic_perturb(df: pd.DataFrame, perturb_ratio: float = 0.5,
                      text_label: str = 'reviewText',
                      type_label: str = 'perturbType',
                      perturb_text_label: str = 'perturbedText',
                      counterfactual_text_label: str = 'counterFactText'):
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
    N_V_Z1 = (df[Z_label] & df[V_label]>0).sum()
    P_V_Z1 = N_V_Z1 / N_Z1

    N_Z0 = (~df[Z_label]).sum()
    N_V_Z0 = (~df[Z_label] & df[V_label]>0).sum()
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