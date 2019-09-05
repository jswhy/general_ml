def get_feb(i):
    if 0 == i:
        return 1
    elif 1 == i:
        return 1
    return get_feb(i-1)+get_feb(i-2)

def get_step(i):
    if 1 == i:
        return 1
    elif 2 == i:
        return 2
    elif 0 == i:
        return 0
    return get_step(i - 1) + get_step(i - 2)


def get_step_list(num):
    list=[]
    for i in range(num):
        list.append(i)
    list[0] = 1
    list[1] = 2
    for i in range(num):
        if 0 == i or 1 == i:
            pass
        else:
            list[i] = list[i-1] + list[i-2]
    return list[num-1]

def if_test(a):
    if a < 1 :
        print(a ," < 1" )
    elif a < 2:
        print(a ," < 2" )

print(get_step(20))
print(get_step_list(20))
if_test(0)
