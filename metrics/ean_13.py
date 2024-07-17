import numpy as np

def check_EAN_13(barre_code):
    numbers = [int(num) for num in barre_code]
    weights = [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3]

    num_arr = np.array(numbers[0:12])
    wei_arr = np.array(weights)

    check_sum = np.dot(num_arr, wei_arr)

    if (check_sum % 10):
        multiple_10 = check_sum + (10 - check_sum % 10)
    else:
        multiple_10 = check_sum

    check_digit = multiple_10 - check_sum

    if check_digit == numbers[12]:
        return True
    else:
        return False