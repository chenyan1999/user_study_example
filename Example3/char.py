import string

def get_chars():
  letters = string.ascii_letters
  digits = string.digits
  symbols = string.punctuation
  chars = letters + digits + symbols
  
  return chars