from BubbleSort import bubble_sort
from BucketSort import bucket_sort
from CountingSort import counting_sort
from HeapSort import heap_sort
from InsertionSort import insertion_sort
from MergeSort import merge_sort
from QuickSort import quick_sort
from RadixSort import radix_sort
from SelectionSort import selection_sort
from ShellSort import shell_sort

arr = [1, 7, 3, 9, 4, 2, 8, 6, 5, 0]

# Apply each sorting algorithm to the array 'arr'
sorted_arr_bubble = bubble_sort(arr.copy())
sorted_arr_bucket = bucket_sort(arr.copy())
sorted_arr_counting = counting_sort(arr.copy())
sorted_arr_heap = heap_sort(arr.copy())
sorted_arr_insertion = insertion_sort(arr.copy())
sorted_arr_merge = merge_sort(arr.copy())
sorted_arr_quick = quick_sort(arr.copy())
sorted_arr_radix = radix_sort(arr.copy())
sorted_arr_selection = selection_sort(arr.copy())
sorted_arr_shell = shell_sort(arr.copy())

# Use assert statements to check if the sorted arrays are equal
assert sorted_arr_bubble == sorted_arr_bucket == sorted_arr_counting == sorted_arr_heap == sorted_arr_insertion == sorted_arr_merge == sorted_arr_quick == sorted_arr_radix == sorted_arr_selection == sorted_arr_shell

print('Congrats, test passed')