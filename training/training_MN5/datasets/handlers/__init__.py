from torch.utils.data import Dataset


class DatasetHandler(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def collate_fn(self, batch):
        raise NotImplementedError

    def data_collator(self, batch):
        raise NotImplementedError
