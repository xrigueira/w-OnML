from river import stream

from . import base

class labeled_902(base.FileDataset):
    """Water quality date from the 902 station.
    
    The file contains water quality data from the Ebro river in Spain recorded every 15 minutes
    from Jan 1st, 1999 to Dec 31st, 2022. The goal is to implement online learning for anomaly 
    detection and prediction.
    
    References
    ----------
    [1] [SAICA EBRO](https://saica.chebro.es/)"""
    
    def __init__(self):
        super().__init__(
            filename="labeled_902.csv",
            task=base.BINARY_CLF,
            n_features=16,
            n_samples=841538
            )
    
    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="label",
            converters={"year": int,
                        "month": int,
                        "day": int,
                        "hour": int,
                        "minute": int,
                        "second": int,
                        "week": int,
                        "weekOrder": int,
                        "ammonium_902": float,
                        "conductivity_902": float,
                        "dissolved_oxygen_902": float,
                        "nitrates_902": float,
                        "ph_902": float,
                        "turbidity_902": float,
                        "water_temperature_902": float,
                        "label": int,
                
            },
            parse_dates={"date": "%Y-%m-%d %H:%M:%S"},
        )