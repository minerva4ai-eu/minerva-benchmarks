from typing import TYPE_CHECKING

from scripts.shared.datasets.handlers.AlpacaHandler import (
    AlpacaHandler,
    AlpacaRawDataset,
)
from scripts.shared.datasets.handlers.ShareGPTHandler import ShareGPTHandler
from scripts.shared.datasets.handlers.SonnetHandler import SonnetHandler
from scripts.shared.datasets.handlers.SquadV2Handler import (
    SquadV2Handler,
    SquadV2RawDataset,
)

if TYPE_CHECKING:
    from scripts.shared.datasets.handlers import DatasetHandler, RawTextDataset

DATASET_HANDLER_MAP: dict[str, type["DatasetHandler"]] = {
    "sharegpt": ShareGPTHandler,
    "sonnet": SonnetHandler,
    "alpaca": AlpacaHandler,
    "squadv2": SquadV2Handler,
}

DATASET_MAP: dict[str, type["RawTextDataset"]] = {
    "alpaca": AlpacaRawDataset,
    "squadv2": SquadV2RawDataset,
}
