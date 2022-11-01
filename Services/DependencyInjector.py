from msilib.schema import Control
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from Data.Repository  import Repository


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    repository = providers.Singleton(
        Repository,
        
    )
    