from random import shuffle
items = [11,21,31]
labels=[1,2,3]
all = list(zip(items,labels))
#print(all)
shuffle(all)
a,b = tuple(zip(*all))

print (a)
print(b)