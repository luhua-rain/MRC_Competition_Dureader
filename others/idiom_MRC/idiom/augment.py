import random
from random import shuffle
random.seed(42)

########################################################################
# Stop_words
########################################################################

stop_words = []
with open('stop_words.json', 'r', encoding='utf-8') as f:
    for i in f.readlines():
        stop_words.append(i.strip())
# print(stop_words)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        if word == 103:
            new_words.append(word)
            continue

        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        print('### delete error ###')
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def swap_word(new_words):

    random_idx_1 = random.randint(0, len(new_words)-1)
    while new_words[random_idx_1] == 103:
        random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1

    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        while new_words[random_idx_2] == 103:
            random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words

    if new_words[random_idx_1] == 103 or new_words[random_idx_2] == 103:
        print('#### error ####')
        return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

def random_swap(words, p):

    n = max(1, int(len(words) * p))
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def eda(words, num_aug):

    augmented_sentences = []
    for _ in range(num_aug):
        a_words = random_swap(words, p=0.2)
        augmented_sentences.append(a_words)

    for _ in range(num_aug):
        a_words = random_deletion(words, p=0.2)
        augmented_sentences.append(a_words)

    shuffle(augmented_sentences)
    augmented_sentences = augmented_sentences[:num_aug]
    augmented_sentences.append(words)

    return augmented_sentences

if __name__ == "__main__":
    words = list('我来到你的城市,在北京')
    print(eda(words, num_aug=3))
