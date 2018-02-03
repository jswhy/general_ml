def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    if 1 >= len(nums):
        return False
    dict = {}
    for i in range(len(nums)):
        if target - nums[i] in dict:
            return [dict[target - nums[i]], i]
        else:
            dict[nums[i]]= i
