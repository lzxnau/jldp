"""
HRRD module for the project HRRD.

Main module for the project HRRD

:Author:  JLDP
:Version: 2024.02.27.01

"""
import datetime
import time
from dateutil import tz
from typing import Tuple
from googleapiclient.discovery import build


class Main:
    """
    Main class for the project HRRD.

    Main class for the project HRRD.

    .. card::
    """

    def __init__(
        self, fstr: str, order: int = 0, gap1: int = 0, gap2: int = 24
    ) -> None:
        """Construct a class instance."""
        api_key = "AIzaSyD9cTuxH_P4bOnaTz0sQIz7l9SGWYOb0sk"
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self.fstr = fstr
        self.order = order
        self.gap1 = gap1
        self.gap2 = gap2

    def search(self) -> None:
        """
        Run a method.

        :param x: Description.
        :type x: None
        :return: None
        :rtype: None
        """
        rlist = []
        uframe = self.UTCTimeframe(self.gap1, self.gap2)
        order = "viewCount" if self.order > 0 else "date"
        request = self.youtube.search().list(
            part="id, snippet",
            q=self.fstr,
            publishedBefore=uframe[0],
            publishedAfter=uframe[1],
            order=order,
            maxResults="20",
            relevanceLanguage="zh-Hans",
            type="video",
        )

        response = request.execute()

        rlist = []
        vids = ""
        for item in response["items"]:
            slist = []
            pt = self.LocalTime(item["snippet"]["publishTime"])
            slist.append("Time: " + pt)
            slist.append("  ID: " + item["id"]["videoId"])
            vids += item["id"]["videoId"] + ","
            slist.append("Chan: " + item["snippet"]["channelTitle"])
            slist.append("Titl: " + item["snippet"]["title"])
            rlist.append(slist)

        vlist = self.videos(vids)
        for i, v in enumerate(vlist):
            rlist[i].extend(v)

        for r in rlist:
            for v in r:
                print(v)
            print("")

    def videos(self, ids: str) -> list:
        """
        Run a method.

        :param x: Description.
        :type x: None
        :return: None
        :rtype: None
        """
        request = self.youtube.videos().list(
            part="contentDetails, statistics, snippet",
            id=ids,
        )

        response = request.execute()
        rlist = []
        for item in response["items"]:
            vlist = []
            vlist.append("Last: " + item["contentDetails"]["duration"])
            vlist.append("View: " + item["statistics"]["viewCount"])
            vlist.append("Like: " + item["statistics"]["likeCount"])
            rlist.append(vlist)

        return rlist

    def UTCTimeframe(self, gap1: int = 0, gap2: int = 24) -> tuple[str]:
        """
        Get an UTC timeframe from local time.

        :param gap1: How many hours before the local time.
        :type gap1: int
        :param gap2: How many hours before the gap1 time.
        :type gap2: int
        :return: The RFC 3339 time format from local time with a gap.
        :rtype: tuple<str>
        """
        ltime = datetime.datetime.now()
        utime = ltime.astimezone(datetime.timezone.utc)
        rtime1 = utime - datetime.timedelta(hours=gap1)
        rtime2 = rtime1 - datetime.timedelta(hours=gap2)
        rstr1 = rtime1.isoformat()
        rstr1 = rstr1[: rstr1.index(".")] + "Z"
        rstr2 = rtime2.isoformat()
        rstr2 = rstr2[: rstr2.index(".")] + "Z"

        return rstr1, rstr2

    def LocalTime(self, ustr: str) -> str:
        """
        Run a method.

        :param x: Description.
        :type x: None
        :return: f
        :rtype: None
        """
        utime = datetime.datetime.strptime(ustr, "%Y-%m-%dT%H:%M:%S%z")
        os = datetime.timedelta(seconds=time.timezone)
        ltime = utime - os
        lstr = ltime.strftime("%Y-%m-%d %H:%M:%S")
        return lstr


if __name__ == "__main__":
    # m = Main("沥心沙大桥", gap1=24 * 2, gap2=24 * 6)
    # m.search()
    m = Main("沥心沙大桥", gap1=24 * 6 + 1, gap2=24 * 2)
    m.search()
