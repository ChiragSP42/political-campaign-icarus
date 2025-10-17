#%%
import pandas as pd

# url = 'https://historical.elections.virginia.gov/elections/search/year_from:2020/year_to:2025/office_id:8'
url = 'https://historical.elections.virginia.gov/elections/search/year_from:1789/year_to:2025'
# url = 'https://www.vpap.org/electionresults/20231107/house/'

tables = pd.read_html(url)
tables[0]
# %%
