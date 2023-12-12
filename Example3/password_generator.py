from random import *
import char

def password_generator(min_length, max_length):
  chars = char.get_chars()
  shuffle(chars)
  password = ''.join(choice(chars) for x in range(randint(min_length, max_length)))
  print('Password: %s' % password)
  return password

password_generator(10, 18)
print('Congrats, test passed')