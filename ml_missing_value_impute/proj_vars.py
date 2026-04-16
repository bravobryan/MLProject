from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

end_date = (dt.now() - relativedelta(days=1)).strftime("%Y-%m-%d")