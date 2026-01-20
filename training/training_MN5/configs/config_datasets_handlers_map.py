from training.training_MN5.datasets.handlers.AlpacaHandler import AlpacaHandler
from training.training_MN5.datasets.handlers.ShareGPTHandler import ShareGPTHandler
from training.training_MN5.datasets.handlers.SonnetHandler import SonnetHandler
from training.training_MN5.datasets.handlers.SquadV2Handler import SquadV2Handler

DATASET_HANDLER_MAP = {
    "sharegpt": ShareGPTHandler,
    "sonnet": SonnetHandler,
    "alpaca": AlpacaHandler,
    "squadv2": SquadV2Handler,
}
