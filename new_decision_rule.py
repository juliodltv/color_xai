import os, sys
sys.dont_write_bytecode = True

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import pickle
from collections import defaultdict

with open("shap_values/shap_values.pkl", "rb") as f:
    combination_spaces = pickle.load(f)

# print(combination_spaces)
# quit()

colors = {
    "blue": 0,
    "yellow": 1,
    "cyan": 2,
    "green": 3,
    "red": 4,
    "magenta": 5,
    "orange": 6,
    "white": 7,
    "black": 8
}

combinations = ['[1]', '[2]', '[3]', '[1 2]', '[1 3]', '[2 3]', '[1 2 3]']
dict_channels = defaultdict(float)


num_channels = 2
for space in combination_spaces.keys():
    for comb in combinations:
        value = 0
        if len(comb) == num_channels+2+(num_channels-1):
            for color, c in colors.items():
                value += max(combination_spaces[space][comb][c].sum(axis=0))
            dict_channels[space + comb] = value

sorted_items_desc = sorted(dict_channels.items(), key=lambda item: item[1], reverse=True)

new_dict = defaultdict(list)
for k, v in sorted_items_desc:
    print(k, f"{v:.4f}")
    key = k.split("[")[0]
    new_dict[key].append(v)

sorted_items_mean = sorted(new_dict.items(), key=lambda item: sum(item[1]), reverse=True)
print("\n\n")
for k, v in sorted_items_mean:
    print(k, f"{sum(v):.4f}")
