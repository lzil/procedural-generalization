import pandas as pd
import matplotlib.pyplot as plt
import argparse
from helpers.utils import filter_csv_pandas

from scipy.stats import pearsonr, spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='fruitbot')
parser.add_argument('--distribution_mode', default='easy')
parser.add_argument('--sequential', default='0', type=int)

# parser.add_argument('--demo_csv', default='demos/demo_infos.csv')
# parser.add_argument('--rm_csv_path', default='reward_models/rm_infos.csv')
parser.add_argument('--rm_csv', default='rm_infos_clean_no_l1.csv')

# use length of demonstration as proxy for demonstration reward
# parser.add_argument('--baseline_reward', action='store_true')

args = parser.parse_args()


# def get_baseline_corrs(demos):
#     pearson_r, pearson_p = pearsonr(demos['length'], demos['returns'])
#     spearman_r, spearman_p = spearmanr(demos['length'], demos['returns'])

#     return (pearson_r, spearman_r)


# all_demos = pd.read_csv(args.demo_csv)
all_RMs = pd.read_csv(args.rm_csv)


reward_constraints = {
    'env_name': args.env_name,
    'mode': args.distribution_mode,
    'sequential': args.sequential
}

# demo_constraints = {
#     'set_name': 'test',
#     'env_name': args.env_name,
#     'mode': args.distribution_mode,
#     'sequential': args.sequential
# }

# filtered_demos = filter_csv_pandas(all_demos, demo_constraints)
filtered_RMs = filter_csv_pandas(all_RMs, reward_constraints)

means = filtered_RMs.groupby('num_dems').mean()
stds = filtered_RMs.groupby('num_dems').std()

print(filtered_RMs) #all fruitbot 
print(means) #mean of all of the ones with each group of dems
print(stds) #std of all of the ones with each group of dems

dp = {}
ds = {}

n = [12, 20, 50, 100, 200]

for i in n:
	is_num_dems = filtered_RMs['num_dems'] == i
	dp[i] = filtered_RMs[is_num_dems]['pearson']
	ds[i] = filtered_RMs[is_num_dems]['spearman']

print(dp[12])

x = n
y = []
for i in n:
	print(i)
	y.append(dp[i])
#y = [dp[12], dp[15], dp[30], dp[50], dp[100], dp[200], dp[300], dp[500], dp[1000]]

for xe, ye in zip(x, y):
	plt.scatter([xe] * len(ye), ye, color = 'blue', label = 'pearson correlation', alpha = 0.4, s = 5)

y = []
for i in n:
	print(i)
	y.append(ds[i])
#y = [ds[12], ds[15], ds[30], ds[50], ds[100], ds[200], ds[300], ds[500], ds[1000]]
for xe, ye in zip(x, y):
	plt.scatter([xe] * len(ye), ye, color = 'orange', label = 'spearman correlation', alpha = 0.4, s = 5)

dpmean = []
dsmean = []
dpstd = []
dsstd = []

for i in n:
	dpmean.append(means['pearson'][i])
	dsmean.append(means['spearman'][i])
	dpstd.append(stds['pearson'][i])
	dsstd.append(stds['spearman'][i])

# plt.errorbar(x, dpmean, yerr=dpstd)
# plt.errorbar(x, dsmean, yerr=dsstd)


plt.plot(x, dsmean, color = 'orange')
plt.plot(x, dpmean, color = 'blue')
plt.scatter(x, dsmean, color = 'orange')
plt.scatter(x, dpmean, color = 'blue')
	
plt.xticks(n)
plt.title('Correlations in FruitBot')
plt.xlabel('demos')
plt.ylabel('r')
#plt.xlim([0, 1050])
plt.show()

#scatter at x=12, 15, 20, 30, 50, 100, 200, 300, 500, 1000 with means and sd's 

# pearson_base, _ = pearsonr(filtered_demos['length'], filtered_demos['return'])
# spearman_base, _ = spearmanr(filtered_demos['length'], filtered_demos['return'])
# ax = means.plot(y=['spearman', 'pearson'], yerr=stds, kind='bar', capsize=4, figsize=(16, 9))
# # plt.axhline(y=pearson_base, label='pearson', ls='--')
# # plt.axhline(y=spearman_base, color='salmon', label='spearman', ls='--')
# plt.ylim(0,1)
# handles, _ = ax.get_legend_handles_labels()
# plt.legend()
# plt.show()
