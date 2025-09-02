import logging
from datetime import date, datetime


class Date(object):
    """Util to handle dates (using ISO 8601 date format)"""

    @staticmethod
    def getFromUserInput(messageToUser):
        userinput = ""
        while userinput == "":
            try:
                userinput = input(messageToUser + "[YYYY-MM-DD] ")
                datetime.strptime(userinput, '%Y-%m-%d')
            except Exception:
                logging.warning("an invalid date was entered: " + userinput)
                userinput = ""
        return userinput

    @staticmethod
    def getTimestamp():
        return datetime.now().strftime("%Y-%m-%d_%H%M%S")

    @staticmethod
    def getTodaysDate():
        return date.today().strftime('%Y-%m-%d')

    @staticmethod
    def getYesterdaysDate():
        return date.today().replace(day=date.today().day-1).strftime('%Y-%m-%d')
