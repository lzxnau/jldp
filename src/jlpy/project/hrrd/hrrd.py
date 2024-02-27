"""
HRRD module for the project HRRD.

Main module for the project HRRD

:Author:  JLDP
:Version: 2024.02.27.01

"""
import datetime
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
        rlist = []
        request = self.youtube.search().list(
            part="id, snippet",
            q="贵州山火",
            publishedAfter="2024-02-20T00:00:00Z",
            order="viewCount",
            maxResults="10",
            relevanceLanguage="zh-Hans",
            type="video, channel",
        )

        response = request.execute()

        rlist = []
        vids = ""
        for item in response["items"]:
            slist = []
            pt = (
                item["snippet"]["publishTime"]
                .replace("T", " ")
                .replace("Z", "")
            )
            slist.append("    Time: " + pt)
            slist.append("      ID: " + item["id"]["videoId"])
            vids += item["id"]["videoId"] + ","
            slist.append(" Channel: " + item["snippet"]["channelTitle"])
            slist.append("   Title: " + item["snippet"]["title"])
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
            vlist.append("Duration: " + item["contentDetails"]["duration"])
            vlist.append("    View: " + item["statistics"]["viewCount"])
            vlist.append("    Like: " + item["statistics"]["likeCount"])
            vlist.append(" Comment: " + item["statistics"]["commentCount"])
            vlist.append("   Audio: " + item["snippet"]["defaultAudioLanguage"])
            vlist.append(" Caption: " + item["contentDetails"]["caption"])
            rlist.append(vlist)

        return rlist

    def UTCTimeframe(self, gap1: int = 0, gap2: int = 24) -> tuple<str>:
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


if __name__ == "__main__":
    m = Main()
    # m.search()
    print(m.UTCTime())
