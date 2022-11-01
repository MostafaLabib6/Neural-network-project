
import Dependency_Injection.DependencyInjector as di
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from GUI.GUI import GUI



@inject
def main(gui: GUI = Provide[di.Container.gui]):
    gui.run()
    print("hello world")

if __name__ == "__main__":
    container = di.Container()
    container.wire(modules=[__name__])
    main()  # <-- dependency is injected automatically
