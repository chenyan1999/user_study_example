def selection_sort(arr):
    for i in range(arr):
        min_index = i
        for j in range(i+1, arr):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr