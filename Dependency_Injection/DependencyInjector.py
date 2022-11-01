from msilib.schema import Control
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from Data.Repository  import Repository
import GUI.GUI as gui
import Services.DataPreProcessingService as dpp
import GUI.Controllers.MainController as mc

class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    repository = providers.Singleton(
        Repository,
        
    )
    service = providers.Factory(
       dpp.DataPreProcessingService
     )
    controlls = providers.Factory(
        mc.MainController,
        
    )
    gui = providers.Factory(
        
        gui.GUI
    )
class Gui(containers.DeclarativeContainer):

    config = providers.Configuration()

    repository = providers.Singleton(
        Repository,
        
    )
    service = providers.Factory(
       dpp.DataPreProcessingService
     )
    controlls = providers.Factory(
        mc.MainController,
        
    )
    gui = providers.Factory(
        
        gui.GUI
    )
  
class Contolers(containers.DeclarativeContainer):

    config = providers.Configuration()

    repository = providers.Singleton(
        Repository,
        
    )
    service = providers.Factory(
       dpp.DataPreProcessingService
     )
    controlls = providers.Factory(
        mc.MainController,
        
    )
    gui = providers.Factory(
        
        gui.GUI
    )
    
class services(containers.DeclarativeContainer):

    config = providers.Configuration()

    repository = providers.Singleton(
        Repository,
        
    )
    service = providers.Factory(
       dpp.DataPreProcessingService
     )
    controlls = providers.Factory(
        mc.MainController,
        
    )
    gui = providers.Factory(
        
        gui.GUI
    )