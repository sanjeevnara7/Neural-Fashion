import os
import numpy as np
import pandas as pd

shape_file = './labels/shape/shape_anno_all.txt'
fabric_file = './labels/texture/fabric_ann.txt'
pattern_file = './labels/texture/pattern_ann.txt'

shape_ann_data = pd.read_csv(shape_file, sep=' ').to_numpy()
fabric_ann_data = pd.read_csv(fabric_file, sep=' ').to_numpy()
pattern_ann_data = pd.read_csv(pattern_file, sep=' ').to_numpy()

img_names = list(shape_ann_data[:, 0])
fabric_ann_data = fabric_ann_data[np.isin(fabric_ann_data[:, 0], img_names)]
pattern_ann_data = pattern_ann_data[np.isin(pattern_ann_data[:, 0], img_names)]

print(fabric_ann_data.shape)
print(pattern_ann_data.shape)
print((shape_ann_data[:, 0] == fabric_ann_data[:, 0]).all(), (pattern_ann_data[:, 0] == fabric_ann_data[:, 0]).all())

full_ann_data = np.concatenate((shape_ann_data, fabric_ann_data[:, 1:], pattern_ann_data[:, 1:]), axis = 1)

#Shuffle the full dataset
np.random.seed(1542)
np.random.shuffle(full_ann_data)
orig_full_ann_data = np.copy(full_ann_data)

attribute_classes = [
    6, 5, 4, 3, 5, 3, 3, 3, 5, 8, 3, 3, #Shape Attributes
    8, 8, 8, #Fabric Attributes
    8, 8, 8 #Color Attributes
]

#Convert the sub classes into 98-index based values
for i in range(full_ann_data.shape[0]):
    full_count = 0
    for idx, count in enumerate(attribute_classes):
        full_ann_data[i, idx+1] += full_count
        full_count += count

completed_keys = []
val_ann_data = []

print("Validation length: ", len(val_ann_data))
#print("Shape: ", full_ann_data.shape)

#Split data based on lowest count subclass till the validation samples reach a number
while(len(val_ann_data) < 4000):
    full_ann_data_list = []
    full_ann_data_dict = {}

    for j in range(100):
        full_ann_data_dict[j] = 0

    for i in range(full_ann_data.shape[0]):
        for j in range(1, 19):
            full_ann_data_dict[int(full_ann_data[i, j])] += 1

    for k,v in full_ann_data_dict.items():
        full_ann_data_list.append([k, v])

    full_ann_data_list = sorted(full_ann_data_list, key=lambda x: x[1])
    #print(full_ann_data_list)

    key = full_ann_data_list[0][0]
    count = 0
    while(str(key) in completed_keys):
        count += 1
        key = full_ann_data_list[count][0]
       
    val_count = full_ann_data_list[count][1] - int(full_ann_data_list[count][1]*0.9)
    curr_count = 0
    completed_full_ann_data = []
    for i in range(full_ann_data.shape[0]):
        if str(key) in list(full_ann_data[i]):
            val_ann_data.append(list(full_ann_data[i]))
            completed_full_ann_data.append(list(full_ann_data[i]))
            curr_count += 1
        
        if curr_count == val_count:
            completed_keys.append(str(key))
            break
    
    new_full_ann_data_list = full_ann_data.tolist()
    new_full_ann_data_list = [x for x in new_full_ann_data_list if x not in completed_full_ann_data]
    full_ann_data = np.array(new_full_ann_data_list)
    #print("After: ", full_ann_data.shape)
    print("Validation length: ", len(val_ann_data))

val_ann_data = np.array(val_ann_data)

train_img_names = list(full_ann_data[:, 0])
val_img_names = list(val_ann_data[:, 0])

train_ann_data = orig_full_ann_data[np.isin(orig_full_ann_data[:, 0], train_img_names)]
valid_ann_data = orig_full_ann_data[np.isin(orig_full_ann_data[:, 0], val_img_names)]

train_ann_data_list = []
train_ann_data_dict = {}

for j in range(100):
    train_ann_data_dict[j] = 0

for i in range(full_ann_data.shape[0]):
    full_count = 0
    for j in range(1, 19):
        train_ann_data_dict[int(full_ann_data[i, j])] += 1

for k,v in train_ann_data_dict.items():
    train_ann_data_list.append([k, v])

train_ann_data_list = sorted(train_ann_data_list, key=lambda x: x[0])

val_ann_data_list = []
val_ann_data_dict = {}

for j in range(100):
    val_ann_data_dict[j] = 0

for i in range(val_ann_data.shape[0]):
    full_count = 0
    for j in range(1, 19):
        val_ann_data_dict[int(val_ann_data[i, j])] += 1
        
for k,v in val_ann_data_dict.items():
    val_ann_data_list.append([k, v])

val_ann_data_list = sorted(val_ann_data_list, key=lambda x: x[0])

#Print the ratio of occurences of examples of a particular subclass in train and validation datasets
ratio_list = []
for i in range(len(val_ann_data_list)):
    if val_ann_data_list[i][1]:
        ratio_value = train_ann_data_list[i][1]/val_ann_data_list[i][1]
        ratio_list.append(ratio_value)
        #print(val_ann_data_list[i][0], val_ann_data_list[i][1], train_ann_data_list[i][0], train_ann_data_list[i][1], ratio_value)
    #else:
        #print(val_ann_data_list[i][0], val_ann_data_list[i][1], train_ann_data_list[i][0], train_ann_data_list[i][1])
    
ratio_list.sort()
print(ratio_list)

#Check WOMEN and MEN count in train and validation datasets
women_count = 0
men_count = 0
for i in range(train_ann_data.shape[0]):
    if train_ann_data[i][0].find("WOMEN") != -1:
        women_count += 1
    else:
        men_count += 1

print("Train data women: ", women_count)
print("Train data men: ", men_count)

women_count = 0
men_count = 0
for i in range(valid_ann_data.shape[0]):
    if valid_ann_data[i][0].find("WOMEN") != -1:
        women_count += 1
    else:
        men_count += 1

print("Validation data women: ", women_count)
print("Validation data men: ", men_count)

print(train_ann_data.shape)
print(train_ann_data[:5])
print(train_ann_data[-5:])

print(valid_ann_data.shape)
print(valid_ann_data[:5])
print(valid_ann_data[-5:])

#Check if any overlap between train and validation images
common_names = [x for x in val_img_names if x in train_img_names]
print(common_names)

np.save('train_data.npy', train_ann_data)
np.save('validation_data.npy', valid_ann_data)