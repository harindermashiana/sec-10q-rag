from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import Dict, Tuple


SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSION_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


@dataclass(frozen=True)
class SecClient:
    headers: Dict[str, str]

    def ticker_to_cik(self, ticker: str) -> str:
        """
        Convert ticker -> zero-padded CIK (10 digits).
        """
        resp = requests.get(SEC_TICKER_CIK_URL, headers=self.headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        t = ticker.upper()
        for item in data.values():
            if item.get("ticker") == t:
                return str(item["cik_str"]).zfill(10)
        raise ValueError(f"Ticker not found: {ticker}")

    @staticmethod
    def quarter_to_month_range(quarter: str) -> Tuple[int, int]:
        mapping = {"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}
        if quarter not in mapping:
            raise ValueError("quarter must be one of: Q1, Q2, Q3, Q4")
        return mapping[quarter]

    def get_10q_filing_url(self, cik: str, year: int, quarter: str) -> str:
        """
        Find the primaryDocument URL for a 10-Q within a year+quarter.
        """
        url = SEC_SUBMISSION_URL.format(cik=cik)
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        filings = data["filings"]["recent"]
        start_m, end_m = self.quarter_to_month_range(quarter)

        for form, date, acc, doc in zip(
            filings["form"],
            filings["filingDate"],
            filings["accessionNumber"],
            filings["primaryDocument"],
        ):
            y, m, _ = map(int, date.split("-"))
            if form == "10-Q" and y == year and start_m <= m <= end_m:
                acc_no_dashes = acc.replace("-", "")
                return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_dashes}/{doc}"

        raise ValueError(f"10-Q not found for CIK={cik}, year={year}, quarter={quarter}")

    def fetch_html(self, url: str) -> str:
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.text
