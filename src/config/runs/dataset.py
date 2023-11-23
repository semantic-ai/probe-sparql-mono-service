
from .constant import CONFIG_PREFIX
from ..base import Settings
from ...enums import DatasetType


class DatasetConfig(Settings):
    type: DatasetType = DatasetType.SINGLE_TOP_LEVEL_ALL_BASED.value
    use_predefined_split: bool = True
    train_test_split: float = 0.2
    tokenize: bool = False
    apply_mlb: bool = True

    predefined_uris: list[str] = [
        'https://data.gent.be/id/besluiten/20.1119.1763.4055',
        'https://data.gent.be/id/besluiten/23.0512.5513.6316',
        'https://data.gent.be/id/besluiten/23.0125.8979.2476',
        'https://data.gent.be/id/besluiten/23.0509.4023.1085',
        'https://data.gent.be/id/besluiten/23.0127.8083.5174',
        'https://data.gent.be/id/besluiten/22.0603.3792.2151',
        'https://data.gent.be/id/besluiten/23.0111.3606.7662',
        'https://data.gent.be/id/besluiten/22.0901.1594.5964',
        'https://data.gent.be/id/besluiten/22.0808.1843.5384',
        'https://data.gent.be/id/besluiten/23.0523.4926.9433',
        'https://data.gent.be/id/besluiten/22.0519.2272.3562',
        'https://data.gent.be/id/besluiten/22.0608.1690.0485',
        'https://data.gent.be/id/besluiten/22.0519.4683.1717',
        'https://data.gent.be/id/besluiten/22.1017.2390.2900',
        'https://data.gent.be/id/besluiten/22.1208.2468.8829',
        'https://data.gent.be/id/besluiten/20.1029.1895.9869',
        'https://data.gent.be/id/besluiten/23.0515.8364.4438',
        'https://data.gent.be/id/besluiten/22.0321.2850.1973',
        'https://data.gent.be/id/besluiten/21.0823.3036.8764',
        'https://data.gent.be/id/besluiten/21.0512.8801.9000',
        'https://data.gent.be/id/besluiten/23.0125.4746.4500',
        'https://data.gent.be/id/besluiten/23.0306.0269.9582',
        'https://data.gent.be/id/besluiten/22.0817.9535.4247',
        'https://data.gent.be/id/besluiten/22.0128.2541.4183',
        'https://data.gent.be/id/besluiten/21.0409.3174.9075',
        'https://data.gent.be/id/besluiten/21.1005.5860.2859',
        'https://data.gent.be/id/besluiten/21.1129.1832.5724',
        'https://data.gent.be/id/besluiten/23.0602.7945.9202',
        'https://data.gent.be/id/besluiten/22.0425.6931.0953',
        'https://data.gent.be/id/besluiten/21.1124.8556.3633',
        'https://data.gent.be/id/besluiten/21.0126.4791.0976',
        'https://data.gent.be/id/besluiten/21.1125.1739.0370',
        'https://data.gent.be/id/besluiten/23.0306.0340.2256',
        'https://data.gent.be/id/besluiten/23.0612.7896.9918',
        'https://data.gent.be/id/besluiten/21.0527.0671.4273',
        'https://data.gent.be/id/besluiten/21.0823.6389.4566',
        'https://data.gent.be/id/besluiten/21.0527.2179.0939',
        'https://data.gent.be/id/besluiten/21.0824.5458.6389',
        'https://data.gent.be/id/besluiten/21.1019.1791.4354',
        'https://data.gent.be/id/besluiten/22.0322.1955.1501',
        'https://data.gent.be/id/besluiten/22.0322.8787.2291',
        'https://data.gent.be/id/besluiten/21.1116.3755.9845',
        'https://data.gent.be/id/besluiten/21.1103.3781.5990',
        'https://data.gent.be/id/besluiten/22.0419.5359.0523',
        'https://data.gent.be/id/besluiten/21.0128.3410.7917',
        'https://data.gent.be/id/besluiten/22.1110.4869.8654',
        'https://data.gent.be/id/besluiten/21.0422.7718.9797',
        'https://data.gent.be/id/besluiten/21.0107.2995.2820',
        'https://data.gent.be/id/besluiten/21.0422.6868.6723',
        'https://data.gent.be/id/besluiten/22.0905.6980.3245',
        'https://data.gent.be/id/besluiten/22.0803.8521.7432',
        'https://data.gent.be/id/besluiten/21.0107.8735.3509',
        'https://data.gent.be/id/besluiten/22.0301.0258.4264',
        'https://data.gent.be/id/besluiten/21.0128.8346.9817',
        'https://data.gent.be/id/besluiten/21.0506.2713.2652',
        'https://data.gent.be/id/besluiten/20.1126.1012.6606',
        'https://data.gent.be/id/besluiten/21.0823.5789.5424',
        'https://data.gent.be/id/besluiten/22.0426.6801.9275',
        'https://data.gent.be/id/besluiten/22.0824.0230.0689',
        'https://data.gent.be/id/besluiten/22.0404.4161.8378',
        'https://data.gent.be/id/besluiten/21.0603.0095.0799',
        'https://data.gent.be/id/besluiten/22.0613.5597.9684',
        'https://data.gent.be/id/besluiten/22.0425.4256.1721',
        'https://data.gent.be/id/besluiten/21.0527.8171.8911',
        'https://data.gent.be/id/besluiten/23.0310.2221.5725',
        'https://data.gent.be/id/besluiten/22.1118.7244.6710',
        'https://data.gent.be/id/besluiten/20.1119.0235.0033',
        'https://data.gent.be/id/besluiten/22.0322.4766.3091',
        'https://data.gent.be/id/besluiten/23.0215.5355.8304',
        'https://data.gent.be/id/besluiten/23.0315.7224.2861',
        'https://data.gent.be/id/besluiten/21.0415.3446.4595',
        'https://data.gent.be/id/besluiten/20.1119.1769.5247',
        'https://data.gent.be/id/besluiten/22.0627.0577.5063'
    ]

    class Config():
        env_prefix = f"{CONFIG_PREFIX}dataset_"
