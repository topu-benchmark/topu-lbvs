from models.gnn_common import BaseGNNModel


class GAT(BaseGNNModel):
    def __init__(self, heads: int = 4, **kwargs):
        super().__init__(
            model_name="gat",
            backbone="gat",
            with_fp=False,
            heads=heads,
            **kwargs,
        )


class GATFP(BaseGNNModel):
    def __init__(self, heads: int = 4, **kwargs):
        super().__init__(
            model_name="gatfp",
            backbone="gat",
            with_fp=True,
            heads=heads,
            **kwargs,
        )