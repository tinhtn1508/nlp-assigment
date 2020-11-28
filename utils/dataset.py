import torch.utils.data as data
import torch
import utils
import pickle

def default_loader(data_path: str):
    return open(data_path, 'r')


class SequenceDataset(data.Dataset):
    def __init__(self, data_path, vocab_path, loader = default_loader):
        self.root_dataset = loader(data_path)
        self.vocab_info = pickle.load(open(vocab_path, 'rb+'))
        self.total_number_sequences = utils.getNumberOfLine(data_path)
        self.total_sequences = self.root_dataset.readlines()

    def __del__(self) -> None:
        self.root_dataset.close()

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        assert index < self.total_number_sequences
        encode_data = self._transform(self.total_sequences[index])
        return encode_data[:,:-1], encode_data[:,-1]

    # def test(self, index):
    #     return self.__getitem__(index)

    def __len__(self) -> int:
        return self.total_number_sequences

    def _transform(self, raw_line: str) -> torch.Tensor:
        encode_value = []
        for char in raw_line[:len(raw_line) - 1]:
            encode_value.append(self.vocab_info[char])
        return torch.Tensor([encode_value])
