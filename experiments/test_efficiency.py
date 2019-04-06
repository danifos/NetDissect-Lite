from time import time, sleep
import sys
import os

LOG_INTERVAL = 5
opt = int(sys.argv[1])

def plot():
  import numpy as np
  import matplotlib.pyplot as plt

  with open('experiments/logs/util.log', 'r') as fi:
    util = np.array([int(a.split()[0]) for a in fi.readlines()])
  with open('experiments/logs/mem.log', 'r') as fi:
    mem = np.array([int(a.split()[0]) for a in fi.readlines()])
  with open('experiments/logs/time.log', 'r') as fi:
    time = np.array([int(a.split()[0]) for a in fi.readlines()])
  time -= time[0]
  time = (time * max(util.shape[0], mem.shape[0])*LOG_INTERVAL / time[-1]).astype(np.int64)
  last = min(int(time[-1]//LOG_INTERVAL), util.shape[0], mem.shape[0])
  x = np.linspace(0, (last+1)*LOG_INTERVAL, last)

  fig, ax1 = plt.subplots()

  ax1.set_xlabel('time (s)')
  color = '#1f77b4'
  ax1.plot(x, util[:last], color=color)[0]
  ax1.set_ylabel('GPU Util (%)', color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax2 = ax1.twinx()
  color = '#ff7f0e'
  ax2.plot(x, mem[:last], color=color)[0]
  ax2.set_ylabel('GPU Memory Usage (MB)', color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()

  ymax = (ax1.get_ylim()[1], ax2.get_ylim()[1])
  for t in time[1:-1]:
    plt.plot((t,t), (0, max(ymax)), '--', color='grey')

  texts = ['feature extraction',
           'calculating threshold',
           'calculating IoU scores',
           'generating results']
  colors = ['yellow', 'blue', 'green', 'red']
  for t1, t2, c, t in zip(time[:-1], time[1:], colors, texts):
    ax1.axvspan(t1, t2, alpha=0.5, color=c)
    ax2.text((t1+t2)/2-30, ymax[1]/1.4, t, rotation=90, va='top', ha='center', color='black', wrap=True)

  for ax, ym in zip((ax1, ax2), ymax):
    ax.set_xlim((0,time[-1]))
    ax.set_ylim((0,ym))
  plt.show()


def main():
  log = open('experiments/logs/time.log', 'w')
  printt = lambda : log.write('%d\n' % time())
  printt()

  import settings
  from loader.model_loader import loadmodel
  from feature_operation import hook_feature,FeatureOperator
  from visualize.report import generate_html_summary
  from util.clean import clean

  fo = FeatureOperator()
  model = loadmodel(hook_feature)

  ############ STEP 1: feature extraction ###############
  features, maxfeature = fo.feature_extraction(model=model)

  for layer_id,layer in enumerate(settings.FEATURE_NAMES):
      printt()
  ############ STEP 2: calculating threshold ############
      thresholds = fo.quantile_threshold(features[layer_id],savepath="quantile.npy")

      printt()
  ############ STEP 3: calculating IoU scores ###########
      tally_result = fo.tally(features[layer_id],thresholds,savepath="tally.csv")

      printt()
  ############ STEP 4: generating results ###############
      generate_html_summary(fo.data, layer,
                            tally_result=tally_result,
                            maxfeature=maxfeature[layer_id],
                            features=features[layer_id],
                            thresholds=thresholds)
      printt()
      if settings.CLEAN:
          clean()
  log.close()

if opt == 0:
  main()
  plot()

elif opt == 1:
  while True:
    os.system('nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader >> experiments/logs/util.log')
    sleep(LOG_INTERVAL)

elif opt == 2:
  while True:
    os.system('nvidia-smi --query-gpu=memory.used --format=csv,noheader >> experiments/logs/mem.log')
    sleep(LOG_INTERVAL)

elif opt == 3:
  plot()
