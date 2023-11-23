from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger

import requests
import pandas as pd

from ..errors import StatuscodeError
from ..enums import EndpointType

import time


class RequestHandler:
    """
    Wrapper around requests library that has some extra functionality that helps to connect with the sparql endpoint
    """

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

    def _internal_request(self, query: str, endpoint: EndpointType):
        """
        Internal request function that has extra error handling

        :param query: query to execute
        :param endpoint: endpoint to communicate with
        :return: the response from the request
        """

        if endpoint is None:
            endpoint = EndpointType.DECISION

        retries = 0
        succes = False

        while not succes:
            r = requests.post(
                EndpointType.match(config=self.config, value=endpoint),
                data={"query": query},
                headers=self.config.request.header,
                auth=(self.config.request.username, self.config.request.password)
            )

            if succes := r.ok:
                self.logger.debug("Status OK")
                return r.json()
            else:
                self.logger.debug(f"Status NOK - retry: {retries}, max_retry: {self.config.request.max_retries}")
                retries += 1
                if retries == self.config.request.max_retries:
                    raise StatuscodeError(r.status_code)

                time.sleep(retries * 10)

    def post2json(self, query: str, endpoint: EndpointType = None):
        """
        Function that takes in the query, executes it and responds with the json output.

        :param query: the query to execute
        :param endpoint: the endpoint to use
        :return: returns the json output
        """
        json_r = self._internal_request(query=query, endpoint=endpoint)
        data = json_r["results"]["bindings"]
        return [{k: v['value'] for k, v in i.items()} for i in data]

    def post2df(self, query: str, endpoint: EndpointType = None) -> pd.DataFrame:
        """
        Funciton that takes in the query, executes it and respons with a dataframe output

        :param query: query to execute
        :param endpoint: the endpoint to use
        :return: the parsed pandas dataframe based on the response
        """
        return pd.DataFrame(self.post2json(query=query, endpoint=endpoint))