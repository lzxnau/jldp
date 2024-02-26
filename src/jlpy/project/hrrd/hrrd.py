"""
HRRD module for the project HRRD.

Main module for the project HRRD

:Author:  JLDP
:Version: 2024.02.27.01

"""
import requests
from googleapiclient.discovery import build


class Main:
    """
    Main class for the project HRRD.

    Main class for the project HRRD.

    .. card::
    """

    def __init__(self) -> None:
        """Construct a class instance."""
        api_key = "AIzaSyD9cTuxH_P4bOnaTz0sQIz7l9SGWYOb0sk"
        self.youtube = build("youtube", "v3", developerKey=api_key)

   def search(self) -> None:
       """
       Run a method.

       :param x: Description.
       :type x: None
       :return: None
       :rtype: None
       """
       request = self.youtube.search().list(
           part="snippet",
           q="Reinforcement Learning"
       )

       response = request.execute()

       for item in response["items"]:
           print(item["snippet"]["title"])


if __name__ == "__main__":
    m = Main()
    m.search
