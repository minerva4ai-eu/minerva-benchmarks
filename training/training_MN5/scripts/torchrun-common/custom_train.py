from transformers import (
    Trainer,
)


class CustomTrainer(Trainer):
    def __init__(self, train_dataloader=None, eval_dataloader=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_train_dataloader = train_dataloader
        self.custom_eval_dataloader = eval_dataloader

    def get_train_dataloader(self):
        if self.custom_train_dataloader is not None:
            return self.custom_train_dataloader
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if self.custom_eval_dataloader is not None:
            return self.custom_eval_dataloader
        return super().get_eval_dataloader(eval_dataset)
