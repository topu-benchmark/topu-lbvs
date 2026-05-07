from models.gnn_common import BaseGNNModel


class GPS(BaseGNNModel):
    def __init__(self, heads: int = 4, **kwargs):
        super().__init__(
            model_name="gps",
            backbone="gps",
            with_fp=False,
            heads=heads,
            **kwargs,
        )


class GPSFP(BaseGNNModel):
    def __init__(self, heads: int = 4, **kwargs):
        super().__init__(
            model_name="gpsfp",
            backbone="gps",
            with_fp=True,
            heads=heads,
            **kwargs,
        )
