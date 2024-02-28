"""
HRRD module for the project HRRD.

Main module for the project HRRD

:Author:  JLDP
:Version: 2024.02.27.01

"""
import datetime
import time
from typing import Tuple
from googleapiclient.discovery import build


class Main:
    """
    Main class for the project HRRD.

    Main class for the project HRRD.

    .. card::
    """

    def __init__(
        self, fstr: str, order: int = 0, gap1: int = 0, gap2: int = 24, stz=0
    ) -> None:
        """Construct a class instance."""
        api_key = "AIzaSyD9cTuxH_P4bOnaTz0sQIz7l9SGWYOb0sk"
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self.fstr = fstr
        self.order = order
        self.gap1 = gap1
        self.gap2 = gap2
        self.stz = int(3600 * stz)
        utc_now = datetime.datetime.utcnow()
        loc_now = datetime.datetime.now()
        self.ltz = int((loc_now - utc_now).total_seconds())

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
            slist.append("Time: " + pt[0])
            slist.append("SrcT: " + pt[1])
            slist.append("  ID: " + item["id"]["videoId"])
            vids += item["id"]["videoId"] + ","
            slist.append("Chan: " + item["snippet"]["channelTitle"])
            slist.append("Titl: " + item["snippet"]["title"])
            rlist.append(slist)

        vlist = self.videos(vids)
        for i, v in enumerate(vlist):
            rlist[i].extend(v)

        for i, r in enumerate(rlist):
            print("  NO: " + str(i + 1))
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

    def LocalTime(self, ustr: str) -> Tuple[str]:
        """
        Run a method.

        :param x: Description.
        :type x: None
        :return: f
        :rtype: None
        """
        utime = datetime.datetime.strptime(ustr, "%Y-%m-%dT%H:%M:%S%z")
        osl = datetime.timedelta(seconds=-self.ltz)
        ost = datetime.timedelta(seconds=-self.stz)
        ltime = utime - osl
        ttime = utime - ost
        lstr = ltime.strftime("%Y-%m-%d %H:%M:%S")
        tstr = ttime.strftime("%Y-%m-%d %H:%M:%S")
        return lstr, tstr

    def checkDS(self, itime: str, dtime: str, ida: bool = True) -> None:
        """
        Run a method.

        :param x: Description.
        :type x: None
        :return: None
        :rtype: None
        """
        pass

    @classmethod
    def initDS(cls, lcode: str = "au", scode: str = "cn") -> None:
        """
        Run a method.

        :param x: Description.
        :type x: None
        :return: None
        :rtype: None
        """
        match lcode:
            case "au":
                cls.ldst = ""
                cls.lida = False
            case _:
                cls.lids = False

        match scode:
            case "cn":
                cls.stz = 3600 * 8
                cls.sids = False
            case _:
                cls.sids = False


if __name__ == "__main__":
    # m = Main("沥心沙大桥", gap1=24 * 2, gap2=24 * 6)
    # m.search()
    m = Main("沥心沙大桥", gap1=24 * 6 + 2, gap2=24 * 2, stz=8)
    m.search()
