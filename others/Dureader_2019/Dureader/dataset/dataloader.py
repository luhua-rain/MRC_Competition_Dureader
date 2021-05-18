import args
import torchtext
from torchtext import data
from torch.utils.data import DataLoader

def x_tokenize(ids):
    return [int(i) for i in ids]
def y_tokenize(y):
    return int(y)

class Dureader():
    def __init__(self, path='./dataset'):

        self.WORD = torchtext.data.Field(batch_first=True, sequential=True, tokenize=x_tokenize,
                               use_vocab=False, pad_token=0)
        self.LABEL = torchtext.data.Field(sequential=False,tokenize=y_tokenize, use_vocab=False)

        dict_fields = {'input_ids': ('input_ids', self.WORD),
                       'input_mask': ('input_mask', self.WORD),
                       'segment_ids': ('segment_ids', self.WORD),
                       'start_position': ('start_position', self.LABEL),
                       'end_position': ('end_position', self.LABEL) }

        self.train, self.dev = torchtext.data.TabularDataset.splits(
                path=path,
                train="train.data",
                validation="dev.data",
                format='json',
                fields=dict_fields)
        self.train_iter, self.dev_iter = torchtext.data.BucketIterator.splits(
                                                                    [self.train, self.dev],  batch_size=args.batch_size,
                                                                    sort_key=lambda x: len(x.input_ids) ,sort_within_batch=True, shuffle=True)
