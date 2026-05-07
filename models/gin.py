from models.gnn_common import BaseGNNModel


class GIN(BaseGNNModel):
    def __init__(self, **kwargs):
        super().__init__(model_name="gin", backbone="gin", with_fp=False, **kwargs)


class GINFP(BaseGNNModel):
    def __init__(self, **kwargs):
        super().__init__(model_name="ginfp", backbone="gin", with_fp=True, **kwargs)