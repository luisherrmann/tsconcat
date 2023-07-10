from addict import Dict


def get_ts_read_config(path: str, driver: str) -> Dict:
    config = Dict(
        {
            "driver": driver,
            "kvstore": {
                "driver": "file",
                "path": path,
            },
        }
    )
    return config
