# config.py
# we define all the configuration here
import pickle

with open('multi_class_dict_1442.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

NB_FAMILIES = 1442
MAX_LEN = 100
TRAIN_BATCH_SIZE = 256 #16
VALID_BATCH_SIZE = 128 #8
EPOCHS = 10
CHAR_INDEX_DICT = {'a': 0,
 'b': 1,
 'c': 2,
 'd': 3,
 'e': 4,
 'f': 5,
 'g': 6,
 'h': 7,
 'i': 8,
 'k': 9,
 'l': 10,
 'm': 11,
 'n': 12,
 'o': 13,
 'p': 14,
 'q': 15,
 'r': 16,
 's': 17,
 't': 18,
 'u': 19,
 'v': 20,
 'w': 21,
 'x': 22,
 'y': 23,
 'z': 24}
MULTI_CLASS_DICT = loaded_dict