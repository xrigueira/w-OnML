from river import stream

from . import base

class labeled_906(base.FileDataset):
    """Water quality date from the 906 station.
    
    The file contains water quality data from the Ebro river in Spain recorded every 15 minutes
    from Jan 1st, 1999 to Dec 31st, 2022. The goal is to implement online learning for anomaly 
    detection and prediction.
    
    References
    ----------
    [1] [SAICA EBRO](https://saica.chebro.es/)"""

    def __init__(self):
        super().__init__(
            filename="labeled_906_cle.csv",
            task=base.BINARY_CLF,
            n_features=15,
            n_samples=631106
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
                        "ammonium_906": float,
                        "conductivity_906": float,
                        "dissolved_oxygen_906": float,
                        "pH_906": float,
                        "turbidity_906": float,
                        "water_temperature_906": float,
                        "label": int,
                
            },
            parse_dates={"date": "%Y-%m-%d %H:%M:%S"},
        )