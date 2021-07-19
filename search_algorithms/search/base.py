import abc

from gamebreaker.search.utils import Edge


class Search(metaclass=abc.ABCMeta):
    def __init__(self, network, state):
        self.classifier = Edge(network, state)

    @abc.abstractmethod
    def search(self):
        pass
