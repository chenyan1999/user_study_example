def bucket_sort(arr):
    # Find the maximum and minimum values in the array
    max_val = max(arr)
    min_val = min(arr)

    # Calculate the range of each bucket
    bucket_range = (max_val - min_val) / len(arr)

    # Create empty buckets
    buckets = [[] for _ in range(arr)]

    # Distribute elements into buckets
    for num in arr:
        index = int((num - min_val) / bucket_range)
        # Handle the case where index equals the length of the buckets list
        if index == len(buckets):
            index -= 1
        buckets[index].append(num)

    # Sort each bucket (using a simple insertion sort in this example)
    for i in range(len(buckets)):
        buckets[i] = sorted(buckets[i])

    # Concatenate the sorted buckets to get the final sorted array
    sorted_array = [num for bucket in buckets for num in bucket]

    return sorted_array