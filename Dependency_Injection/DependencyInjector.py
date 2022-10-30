from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from Data.Repository  import Repository
from GUI.GUI import GUI

class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

   
    gui = providers.Singleton(
        
        GUI
    )
    repository = providers.Singleton(
        Repository,
        
    )
    service = providers.Factory(
       
    )
