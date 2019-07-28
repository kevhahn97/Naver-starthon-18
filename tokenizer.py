import sentencepiece as spm
import nsml
import os

NSML_SESSION = 'team_220/18_tcls_query/58'

class Tokenizer():
    def __init__(self, vocab_size, data_dir, mode):
        self.sp = spm.SentencePieceProcessor()
        self.vocab_size = vocab_size

        if mode == 'train':
            self.data_file_path = os.path.join(data_dir, 'train/train_data/train_data')
            
            def save_and_train_tokenizer(path, *args, **kwargs):
                with open(self.data_file_path) as f:
                    data = f.read().splitlines()
                    corpus = list()
                    for line in data:
                        corpus += line.split("\t")[1:]
                    with open('corpus', 'w') as wf:
                        for text in corpus:
                            wf.write("%s\n" % text)
                templates = '--input={} --model_prefix={} --vocab_size={} --hard_vocab_limit=false'
                spm.SentencePieceTrainer.Train(templates.format('corpus', os.path.join(path, 'SPM'), self.vocab_size))
                self.sp.Load(os.path.join(path, 'SPM.model'))

            nsml.save('tokenizer', save_fn=save_and_train_tokenizer)
            
        elif mode == 'test':
            def load_tokenizer(path, *args, **kwargs):
                self.sp.Load(os.path.join(path, 'SPM.model'))
            nsml.load(checkpoint='tokenizer', load_fn=load_tokenizer, session=NSML_SESSION)    


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
        spm.SentencePieceTrainer.Train(templates.format('corpus', 'SPM', self.vocab_size))


    def text2ids(self, text):
        return self.sp.EncodeAsIds(text)