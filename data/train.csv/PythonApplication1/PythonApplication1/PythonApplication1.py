import csv
data_list = []
data2_list = []
with open("train.csv", "r", encoding='UTF-8') as f:
    lines = f.readlines()
    for line in lines:
        a = line.split(",")
        if a[2][0:1] == '@':
            continue
        elif a[2][0:1] == '"':
            continue
        elif a[2][0:1] == '-':
            continue
        elif a[2][0:1] == '#':
            continue
        else:
            data_list.append(a)
            if a[2][0:1] == '@':
                continue
            elif a[2][0:1] == '"':
                continue
            elif a[2][0:1] == '-':
                continue
            elif a[2][0:1] == '#':
                continue
            else:
                data2_list.append(a)


print(len(data2_list))
