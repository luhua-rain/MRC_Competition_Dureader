from parameters import *

stop_words = {'我', '我的', '我自己', '我们', '我们的', '我们自己', '你',
              '你的', '你自己', '他', '他的', '他自己', '她', '她的', '她自己',
              '它', '它的', '它自己', '他们', '他们的', '他们自己', '什么',
              '哪个', '谁', '这', '那', '这些', '那些', '是', '一个', '这个',
              '并且', '但是', '如果', '或者', '因为', '直到', '当',
              '~', '！', '@', '#', '￥', '%', '……', '&', '*', '（', '）',
              '{', '}', '【', '】', '：', '；', '“', '‘', '《', '》', '，',
              '。', '？', '、', '!', '$', '^', '(', ')', '-', '_', '+',
              '=', '{', '}', '[', ']', ':', ';', '"', "'", '<', ',', '>',
              '.', '?', '/'}

special_words = ['[', ']', 'MASK', 'UNK', '[MASK]', '[UNK]', '#',
                '[MASK', 'MASK]', '[UNK', 'UNK]', 'idiom']


def is_target(word):
    for sp_word in special_words:
        if sp_word in word:
            return True
    return False


#######################################################################
# Randomly delete words from the sentence with probability p
def random_deletion(words, p):
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p or is_target(word):
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


#######################################################################
# get the synonyms 5 words if exists

def get_synonyms(word):
    sy = []
    try:
        for synset in wn.synsets(word, lang='cmn'):
            sy.extend(synset.lemma_names('cmn'))
        return sy
    except:
        return []


def select_word(word):
    if word in stop_words or len(word) < 2 or is_target(word):
        return False
    return True


# Replace n words in the sentence with synonyms from synonyms
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if
                                 (word not in stop_words) and (not is_target(word))]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    return new_words


######################################################################
# Randomly swap two words in the sentence n times
def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random.randint(0, len(new_words) - 1)
    counter = 0
    while random_idx_2 == random_idx_1 or \
            is_target(new_words[random_idx_1]) or \
            is_target(new_words[random_idx_2]):
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


########################################################################
# Randomly insert n words into the sentence
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    while True:
        random_idx = random.randint(0, len(new_words) - 1)
        if not is_target(new_words[random_idx]):
            break
    new_words.insert(random_idx, random_synonym)


########################################################################
# main data augmentation function
def re_bonding(words):
    new_words = []
    idx = 0
    while idx < len(words):
        if words[idx] == '[' and idx + 1 < len(words) \
                and words[idx + 1] == 'UNK' and idx + 2 < len(words) \
                and words[idx + 2] == ']':
            new_words.append('[UNK]')
            idx += 3
        elif words[idx] == '#' and idx + 1 < len(words) \
                and is_target(words[idx + 1]) and idx + 2 < len(words) \
                and words[idx + 2] == '#':
            new_words.append(''.join([words[idx], words[idx + 1], words[idx + 2]]))
            idx += 3
        else:
            new_words.append(words[idx])
            idx += 1
    return new_words


def augment(sentence, alpha_sr=0.15, alpha_ri=0.12, alpha_rs=0.12, p_rd=0.12, is_show=False):
    words = jieba.lcut(sentence, cut_all=False)
    words = [word for word in words if word is not '']
    words = re_bonding(words)
    num_words = len(words)
    p = random.uniform(0, 1)
    # p = 0.9
    if p < 0.2:
        if is_show:
            print("原文")
        return ''.join(('%s' % id for id in words))
    elif p < 0.4:
        n_sr = max(1, int(alpha_sr * num_words))
        if is_show:
            print("随机替换%d个词语" % (n_sr))
        a_words = synonym_replacement(words, n_sr)
        return ''.join(('%s' % id for id in a_words))
    elif p < 0.6:
        n_ri = max(1, int(alpha_ri * num_words))
        if is_show:
            print("随机插入%d个词语" % (n_ri))
        a_words = random_insertion(words, n_ri)
        return ''.join(('%s' % id for id in a_words))
    elif p < 0.8:
        n_rs = max(1, int(alpha_rs * num_words))
        if is_show:
            print("随机交换%d个词语" % (n_rs))
        a_words = random_swap(words, n_rs)
        return ''.join(('%s' % id for id in a_words))
    else:
        a_words = random_deletion(words, p_rd)
        if is_show:
            print("随机以概率%f删除词语" % (p_rd))
        return ''.join(('%s' % id for id in a_words))


if __name__ == '__main__':
    # with open(config.data_root + "split_%s_data.json" % ("test"), "r") as f:
    #     split_data = json.load(f)
    for i in range(30):
        s = "平静却深刻的声音，富有穿透音乐底层的生命力，蔡健雅的歌声犹如一股清流征服了在场的所有观众。来自美国而今在香港#idiom517740#的DJGruv以他对R&B,Trance等各种音乐元素的跨界理解，玩转混音界，与轩尼诗炫音之乐的混搭理念[UNK]。欧洲顶尖的电音组合Sylver则以他们与众不同的流行加电音风格，配和乐队纯净的音色、宽广的音域，加上合成器、电子鼓，吉他、钢琴和其他打击乐器的伴奏，现身演绎轩尼诗VSOP“融汇世界炫亮音乐的体验”这种全新的风尚。"
        print(augment(s, is_show=True))
