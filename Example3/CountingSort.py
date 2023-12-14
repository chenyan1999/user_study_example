def counting_sort(arr):
    max_val = max(arr) + 1
    count = [0] * max_val
    output = [0] * len(arr)
    for num in arr:
        count[num] += 1
    for i in range(1, max_val):
        count[i] += count[i - 1]
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i]] - 1] = arr[i]
        count[arr[i]] -= 1
    return output
