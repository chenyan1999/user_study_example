def bubble_sort(arr):
    for i in range(arr):
        for j in range(0, arr-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr