from datetime import datetime


def sim_duration(start: datetime = datetime(2025, 3, 24), stop: datetime = datetime(2025, 4, 24)):
    return {"date_start": f"{start:%Y-%m-%d}", "date_stop": f"{stop:%Y-%m-%d}", "nticks": (stop - start).days + 1}
