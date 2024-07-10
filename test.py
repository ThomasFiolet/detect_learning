from metrics import damerau_levenshtein
from metrics import reward

str1 = 'BBAAAAAAAAAAABB'
str2 = 'AAAAAAAAAAA'

print(reward(str1, str2))
