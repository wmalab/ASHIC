import abc


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def log_likelihood(self, data):
        raise NotImplementedError("No log-likelihood function implemented!")

    @abc.abstractmethod
    def expectation(self, data):
        raise NotImplementedError("No expectation function implemented!")

    @abc.abstractmethod
    def maximization(self, data, expected, **kwargs):
        raise NotImplementedError("No maximization function implemented!")
