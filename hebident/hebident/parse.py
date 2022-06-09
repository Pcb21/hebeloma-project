import math


class SBase:

    def __init__(self, db_name):
        self.db_name = db_name

    @staticmethod
    def _subparse(e):
        try:
            return float(e.strip())
        except ValueError:
            return math.nan

    def _parse(self, value):
        # (a;b;c;d)
        v = value.replace("(", "")
        v = v.replace(")", "")
        v = v.split(";")
        return [self._subparse(vi) for vi in v]


class SToSingle(SBase):

    def __init__(self, db_name):
        super().__init__(db_name)

    def parse(self, value):
        sm = 0.0
        cnt = 0
        # Seems possible that value can already be a float by here?
        if isinstance(value, float):
            if math.isnan(value):
                return None
            else:
                return value
        for v in self._parse(value):
            if not math.isnan(v):
                cnt += 1
                sm += v
        if cnt == 0:
            return None
        return sm/float(cnt)


class SToFour(SBase):
    def __init__(self, db_name):
        super().__init__(db_name)

    def parse(self, value):
        elems = self._parse(value)
        return [None if math.isnan(v) else v for v in elems]


def parse_formatted_string(s):
    # Unpacks a string of form:
    # {Europe: 59.65%; Northern America: 35.4%; Asia-Temperate: 4.95%;}
    res = dict()
    s = s.replace("{", "").replace("}", "")
    s = s.split(";")
    for elem in s:
        elem = elem.strip()
        if not elem:
            continue
        key, value = elem.split(":")
        key = key.strip()
        value = value.replace("%", "").strip()
        value = float(value)/100.0
        res[key] = value
    return res
