class Solution:
    def twoSum(self, nums, target):  
        hashmap = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in hashmap:
                return [hashmap[complement], i]
            hashmap[num] = i
        return []  


nums = [2, 7, 11, 15,5]
target = 12

solution = Solution()
result = solution.twoSum(nums, target)  
print(result) 
