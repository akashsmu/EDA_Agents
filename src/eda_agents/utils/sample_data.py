import numpy as np
import pandas as pd


def make_sample_churn_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic telecom churn dataset with realistic signal and deliberate quality problems
    (missing values, duplicates, inconsistent labels, numbers stored as text, outliers),
    so every EDA tool has something to find.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
    monthly = np.round(20 + 45 * (internet == "Fiber optic") + 25 * (internet == "DSL") + rng.normal(10, 8, n), 2)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.1, n), 2)

    # Churn driven by short tenure, month-to-month contracts and fiber pricing.
    logit = -1.0 - 0.05 * tenure + 1.6 * (contract == "Month-to-month") + 0.7 * (internet == "Fiber optic")
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHJKLMNP'), 5))}" for _ in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": (rng.random(n) < 0.16).astype(int),
        "tenure": tenure,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "MonthlyCharges": monthly,
        "TotalCharges": total.astype(str),  # stored as text, like the real Telco dataset
        "SupportCalls": rng.poisson(1.5 + 1.5 * (churn == "Yes")),
        "Churn": churn,
    })

    # Data quality problems
    df.loc[rng.choice(n, int(n * 0.06), replace=False), "MonthlyCharges"] = np.nan
    df.loc[rng.choice(n, int(n * 0.03), replace=False), "PaymentMethod"] = None
    df.loc[rng.choice(n, 15, replace=False), "Contract"] = "month-to-month "
    df.loc[rng.choice(n, 5, replace=False), "SupportCalls"] = rng.integers(40, 60, 5)
    df = pd.concat([df, df.sample(12, random_state=seed)], ignore_index=True)
    return df
