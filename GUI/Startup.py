
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject


@inject
def main(service: Service = Provide[Container.service]) -> None:
    print("hello world")

if __name__ == "__main__":
    container = Container()
    container.wire(modules=[__name__])

    main()  # <-- dependency is injected automatically
