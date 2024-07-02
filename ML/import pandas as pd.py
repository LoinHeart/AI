import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate random data
np.random.seed(0)
n_samples = 1000

# Generate random dates spanning several years
start_date = datetime(2018, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = [start_date + timedelta(days=np.random.randint((end_date - start_date).days)) for _ in range(n_samples)]

# Generate random prices
prices = np.random.randint(30000, 80000, size=n_samples)

# Generate random locations
locations = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Miami', 'San Francisco'], size=n_samples)

# Generate random selling rates
selling_rates = np.random.uniform(0.5, 0.9, size=n_samples)

# Create DataFrame
selling_data = pd.DataFrame({
    'timestamp': date_range,
    'price': prices,
    'location': locations,
    'selling_rate': selling_rates
})

# Save to CSV
selling_data.to_csv('selling_data.csv', index=False)
