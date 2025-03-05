import sys


def sort_1(my_list):
    list1 = sorted(my_list)
    return list1


def sort_2(my_list):
    sorted_list2 = []
    length = len(my_list)
    for i in range(length):
        sorted_list2.append(min(my_list))
        my_list.remove(min(my_list))
    return sorted_list2


if __name__ == '__main__':
    input = sys.argv[1]
    my_list = []
    for list in range(len(input)):
        my_list.append(int(input[list]))

    sorted_list1 = sort_1(my_list)
    sorted_list2 = sort_2(my_list)
