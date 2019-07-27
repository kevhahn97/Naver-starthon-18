import sentencepiece as spm
import os

class Tokenizer():
    def __init__(self, vocab_size, data_dir):
        # Train SP Tokenizer
        self.prefix = 'SPM'
        self.vocab_size = vocab_size
        self.data_file_path = os.path.join(data_dir, 'train/train_data/train_data')
        self.train_tokenizer()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load('{}.model'.format(self.prefix))


    def train_tokenizer(self):
        with open(self.data_file_path) as f:
            data = f.read().splitlines()
            corpus = list()
            for line in data:
                corpus += line.split("\t")[1:]
            with open('corpus', 'w') as wf:
                for text in corpus:
                    wf.write("%s\n" % text)
        templates = '--input={} --model_prefix={} --vocab_size={} --hard_vocab_limit=false'
        spm.SentencePieceTrainer.Train(templates.format('corpus', self.prefix, self.vocab_size))


    def text2ids(self, text):
        return self.sp.EncodeAsIds(text)