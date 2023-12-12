import string

def get_chars():  
  letters = [c for c in string.ascii_letters]
  digits = [c for c in string.digits]
  symbols = [c for c in string.punctuation]
  chars = letters + digits + symbols
  
  return chars