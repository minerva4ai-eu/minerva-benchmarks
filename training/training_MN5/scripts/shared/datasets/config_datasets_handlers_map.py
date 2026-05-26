from typing import TYPE_CHECKING

from shared.datasets.handlers.AlpacaHandler import AlpacaHandler
from shared.datasets.handlers.ShareGPTHandler import ShareGPTHandler
from shared.datasets.handlers.SonnetHandler import SonnetHandler
from shared.datasets.handlers.SquadV2Handler import SquadV2Handler

if TYPE_CHECKING:
    from shared.datasets.handlers import DatasetHandler

DATASET_HANDLER_MAP: dict[str, type["DatasetHandler"]] = {
    "sharegpt": ShareGPTHandler,
    "sonnet": SonnetHandler,
    "alpaca": AlpacaHandler,
    "squadv2": SquadV2Handler,
}
