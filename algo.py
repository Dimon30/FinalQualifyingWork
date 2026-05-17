class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums) - 1
        m = len(nums) // 2
        flag = False
        while l < r:
            if flag:
                if nums[m] == target:
                    return m
                elif nums[m] > target:
                    r = m - 1
                    m = (r + l) // 2
                else:
                    l = m + 1
                    m = (r + l) // 2

            else:
                if nums[m] <= nums[r]:
                    if nums[m] <= target <= nums[r]:
                        flag = True
                        l = m
                        m = (r + l)//2
                    else:
                        r = m
                        m = (r + l)//2
                else:
                    if nums[l] <= target <= nums[m]:
                        flag = True
                        r = m
                        m = (r + l)//2
                    else:
                        l = m
                        m = (r + l)//2
                if nums[m] == target:
                    return m
        return -1

test = Solution()
nums = [4,5,6,7,0,1,2]
target = 3
test.search(nums, target)