# coding = utf-8

nums_origin = [233,333,-99,656,13,5,76,-100,90,0,0,0,0,1,3]

def inser_sort(nums):
    for i in range(len(nums)):
        value = nums[i]
        for j in range(i, -1, -1):
            if nums[j] > value:
                nums[j + 1] = nums[j]
                nums[j] = value
    return nums

b = inser_sort(nums_origin)
print(b)