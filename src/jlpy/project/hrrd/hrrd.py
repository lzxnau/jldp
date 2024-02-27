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

    def UTCTime(self, gap: int = 24) -> str:
        """
        Run a method.

        :param x: Description.
        :type x: None
        :return: None
        :rtype: None
        """
        ltime = datetime.datetime.now()
        utime = ltime.astimezone(datetime.timezone.utc)
        rtime = utime - datetime.timedelta(hours=gap)
        rstr = rtime.isoformat().replace("+00:00", "Z")
        print(ltime)
        print(utime)
        print(rtime)

        return ""


if __name__ == "__main__":
    m = Main()
    # m.search()
    m.UTCTime()
