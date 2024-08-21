from data_initial_inputs import Input, LogisticsParameters
import data_forecast_error_scenarios as error_scen
import data_demand_estimation as demand


class Data:
    """Read all data needed to run models."""
    def __init__(self, args):
        self.args = args
        # Initial inputs
        self.NumericInputs = Input(args=self.args)
        self.NumericInputs.get_all_inputs()
        self.numeric_args = vars(self.NumericInputs)
        self.numeric_args.pop('args')

    def create_data(self):
        """Create instances of choice."""
        # Create logistic parameters
        LogisticsParams = LogisticsParameters(
            args=self.args,
            input_args=self.numeric_args,
            )
        if self.args["data_opt"] in [3, 4]:
            LogisticsParams.get_params(mode='create')
        # Create forecast error data
        FE = error_scen.ForecastError(args=self.numeric_args)
        if self.args["data_opt"] in [1, 4]:
            FE.createFEData(oos=True, args=self.args)
        # Create demand data
        if self.args["data_opt"] in [2, 4]:
            LogisticsParams.get_params(mode='read')
            FE.readFEData(oos_only=False, args=self.args)
            DE = demand.Demand(self.args, FE, LogisticsParams)
            DE.create_all_demand_data()

    def read_data(self):
        # Forecast error data
        FE = error_scen.ForecastError(args=self.numeric_args)
        FE.readFEData(oos_only=False, args=self.args)
        # Logistics parameters
        LogisticsParams = LogisticsParameters(
            args=self.args,
            input_args=self.numeric_args,
            )
        LogisticsParams.get_params(mode='read')
        # Demand data
        self.args['fix_along_err'] = bool(self.args['fix_along_err'])
        DE = demand.Demand(self.args, FE, LogisticsParams)
        DE.read_demand_data()
        # Copy all attributes
        for data_obj in [self.NumericInputs, FE, DE, LogisticsParams]:
            for var, val in vars(data_obj).items():
                try:
                    getattr(self, var)
                except AttributeError:
                    setattr(self, var, val)
