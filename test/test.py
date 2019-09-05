def get_step(target):
    if(2 >= target):
        return target
    return get_step(target-1)+get_step(target-2)

def get_step_back(target):
    list = [1, 2]
    for i in range(target):
        if 1 >= i:
            pass
        else:
            list.append(list[i-1]+list[i-2])
    return list[target-1]

def is_pal(num):
    tem = num
    res = 0
    while 0 != num:
        res = 10 * res + num % 10
        num // 10
        print(res, num)


    return 0 == res - tem
print(is_pal(521))
