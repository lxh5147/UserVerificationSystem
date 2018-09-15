def my_func():
    print(mydict)


def my_func2():
    global mydict
    mydict = dict()
    mydict['hi'] = 'lxh'


my_func2()
my_func()
