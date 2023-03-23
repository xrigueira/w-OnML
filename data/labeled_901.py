from river import stream

from . import base

class labeled_901(base.FileDataset):
    """Water quality date from the 901 station.
    
    The file contains water quality data from the Ebro river in Spain recorded every 15 minutes
    from Jan 1st, 1999 to Dec 31st, 2022. The goal is to implement online learning for anomaly 
    detection and prediction.
    
    References
    ----------
    [1] [SAICA EBRO](https://saica.chebro.es/)"""
    
    def __init__(self):
        super().__init__(
            filename="labeled_901.csv",
            task=base.BINARY_CLF,
            n_features=15,
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
                        "ammonium_901": float,
                        "conductivity_901": float,
                        "dissolved_oxygen_901": float,
                        "ph_901": float,
                        "turbidity_901": float,
                        "water_temperature_901": float,
                        "label": int,
                
            },
            parse_dates={"date": "%Y-%m-%d %H:%M:%S"},
        )