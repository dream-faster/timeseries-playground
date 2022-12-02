from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from utils.market_data import get_market_data
from tqdm import tqdm
from sktime.forecasting.model_selection import ExpandingWindowSplitter
tqdm.pandas()

data = get_market_data()
splitter = ExpandingWindowSplitter(fh=1, initial_window=100, step_length=50)

ba = splitter.split(data)

def train(data):
    return ETSModel(data).fit(maxiter=10000)
a = data['SP500'].expanding(100).progress_apply(train)
