from msilib.schema import Control
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
import Services.DataPreProcessingService as dpp

class Container(containers.DeclarativeContainer):

    config = providers.Configuration()


    service = providers.Factory(
       dpp.DataPreProcessingService
     )
