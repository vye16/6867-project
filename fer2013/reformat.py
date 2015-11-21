train = []
test1 = []
test2 = []

with open("fer2013.csv", 'r') as f:
    hdr = f.readline()
    for line in f.readlines():
        row = line.split(',')
        row[1] = row[1].replace(' ', ',')
        if row[-1] == "Training\n":
            train.append(','.join(row[:-1]))
        elif row[-1] == "PrivateTest\n":
            test2.append(','.join(row[:-1]))
        else:
            test1.append(','.join(row[:-1]))

print "%i training points" % len(train)
print "%i test1 points" % len(test1)
print "%i test2 points" % len(test2)

with open("train.csv", 'w') as f:
    for line in train:
        f.write("%s\n" % line)

with open("test1.csv", 'w') as f:
    for line in test1:
        f.write("%s\n" % line)

with open("test2.csv", 'w') as f:
    for line in test2:
        f.write("%s\n" % line)
