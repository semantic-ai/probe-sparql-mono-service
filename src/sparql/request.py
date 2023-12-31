from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
import traceback
import pandas as pd

from ..errors import StatuscodeError
from ..enums import EndpointType, AuthType

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

        match self.config.request.auth_type:
            case AuthType.NONE:
                self.auth = None
            case AuthType.BASIC:
                self.auth = HTTPBasicAuth(
                    username=self.config.request.username,
                    password=self.config.request.password,
                )
            case AuthType.DIGEST:
                self.auth = HTTPDigestAuth(
                    username=self.config.request.username,
                    password=self.config.request.password,
                )

        self.logger.debug(f"auth type: {self.config.request.auth_type}")

        retries = 1
        succes = False
        last_ex = None

        while not succes:
            # add delay if it is not the first retry
            if retries != 1:
                time.sleep(retries * 10)

            # verify that it is not greater or eqaul to max retries
            if retries >= self.config.request.max_retries:
                self.logger.error(f"Status NOK - retry: {retries}, max_retry: {self.config.request.max_retries}")
                raise Exception(f"Max retries exceeded for sparql request with exception: {last_ex}")

            try:
                r = requests.post(
                    EndpointType.match(config=self.config, value=endpoint),
                    timeout=10,
                    data={"query": query},
                    headers=self.config.request.header,
                    auth=self.auth,
                )

                if succes := r.ok:
                    return r.json()
                else:
                    self.logger.info(f"response: {r.status_code} {r.text}")

            except requests.exceptions.Timeout as ex:
                self.logger.info(f"Request execution timed out: {ex}")
                last_ex = "TIMEOUT"

            except Exception as ex:
                self.logger.info(
                    f"During execution of the request, the following error occured: {traceback.format_exc()}")
                last_ex = ex

            finally:
                retries += 1

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
