from itertools import groupby

def is_palindrome(s):
    return s == s[::-1]

def is_serial(s):
    return all(int(s[i]) == int(s[i-1]) + 1 for i in range(1, len(s))) or all(int(s[i]) == int(s[i-1]) - 1 for i in range(1, len(s)))

def is_double_sequential(s):
    return all(s[i] == s[i + 1] and (int(s[i]) == int(s[i + 2]) + 1 or int(s[i]) == int(s[i + 2]) - 1) for i in range(0, len(s) - 2, 2))

def is_duplicated_and_serial(s):
    return (len(s) % 2 == 0 and 
            all(s[i] == s[i + len(s)//2] for i in range(len(s)//2)) and 
            (is_serial(s[:len(s)//2]) or is_serial(s[len(s)//2:])))

def count_repeats(s):
    return len(s) - len(set(s))

def longest_consecutive_repeats(s):
    return max([len(list(g)) for k, g in groupby(s)])

def lower_number_value(s):
    return sum(10 - int(digit) for digit in s)

def zero_between_features(s):
    return int('0' in s)

def serial_length_value(s):
    max_length = 0
    current_length = 1
    for i in range(1, len(s)):
        if int(s[i]) == int(s[i-1]) + 1 or int(s[i]) == int(s[i-1]) - 1:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 1
    return max_length

def extract_features(plate_number):
    features = {}
    length = len(plate_number)
    if length == 4:
        weight_factor = 6
    elif length == 5:
        weight_factor = 3
    else:
        weight_factor = 1
    
    repeat_count = count_repeats(plate_number)
    four_repeats = 1 if repeat_count >= 4 else 0

    features['num_digits'] = length
    features['num_zeros'] = plate_number.count('0') * weight_factor
    features['num_repeats'] = repeat_count * weight_factor
    features['four_repeats'] = four_repeats * 10
    features['longest_consecutive_repeats'] = longest_consecutive_repeats(plate_number) * weight_factor
    features['palindrome'] = is_palindrome(plate_number) * weight_factor
    if length == 6 and is_palindrome(plate_number):
        features['palindrome'] *= (1/3)
    features['serial'] = is_serial(plate_number) * weight_factor
    features['double_sequential'] = is_double_sequential(plate_number) * weight_factor
    features['duplicated_and_serial'] = is_duplicated_and_serial(plate_number) * weight_factor
    features['repeats_and_serial'] = repeat_count * is_serial(plate_number) * weight_factor
    features['zeros_and_repeats'] = plate_number.count('0') * repeat_count * weight_factor
    features['double_and_repeats'] = is_double_sequential(plate_number) * repeat_count * weight_factor
    features['zeros_and_serial'] = plate_number.count('0') * is_serial(plate_number) * weight_factor
    features['palindrome_and_repeats'] = is_palindrome(plate_number) * repeat_count * weight_factor
    
    features['lower_number_value'] = lower_number_value(plate_number) * weight_factor
    features['zero_between_features'] = zero_between_features(plate_number) * weight_factor
    features['serial_length_value'] = serial_length_value(plate_number) * weight_factor
    
    features['num_repeats_and_length'] = repeat_count * length * weight_factor
    features['serial_and_length'] = is_serial(plate_number) * length * weight_factor
    
    return features
