"""Data contracts (pandera) for the raw feed and master data."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

SHIPMENT_COLUMNS = [
    "shipment_id",
    "site_id",
    "customer_id",
    "ship_date",
    "pallets",
    "temp_zone",
    "order_created_at",
    "ingested_at",
]

TYPED_COLUMNS = {
    "ship_date": "datetime",
    "pallets": "int",
    "order_created_at": "datetime",
    "ingested_at": "datetime",
}


def shipment_schema(
    sites: pd.DataFrame, customers: pd.DataFrame, max_pallets: int
) -> pa.DataFrameSchema:
    """Row-level contract. Every failure here carries a row index and leads to quarantine."""
    pairs = set(zip(customers["site_id"], customers["customer_id"], strict=True))

    def pair_known(df: pd.DataFrame) -> pd.Series:
        keys = pd.Series(list(zip(df["site_id"], df["customer_id"], strict=True)), index=df.index)
        known = df["site_id"].isin(sites["site_id"]) & df["customer_id"].isin(
            customers["customer_id"]
        )
        # only judge the pair when both ids are individually valid
        return ~known | keys.isin(pairs)

    return pa.DataFrameSchema(
        {
            "shipment_id": pa.Column(str, pa.Check.str_matches(r"^SHP\d{9}$"), nullable=False),
            "site_id": pa.Column(str, pa.Check.isin(sites["site_id"].tolist()), nullable=False),
            "customer_id": pa.Column(
                str, pa.Check.isin(customers["customer_id"].tolist()), nullable=False
            ),
            "ship_date": pa.Column("datetime64[ns]", nullable=False),
            "pallets": pa.Column(
                pa.Int64, [pa.Check.gt(0), pa.Check.le(max_pallets)], nullable=False
            ),
            "temp_zone": pa.Column(str, pa.Check.isin(["frozen", "chilled"]), nullable=False),
            "order_created_at": pa.Column("datetime64[ns]", nullable=False),
            "ingested_at": pa.Column("datetime64[ns]", nullable=False),
        },
        checks=[
            pa.Check(pair_known, name="site_customer_pair_known"),
            pa.Check(
                lambda d: d["order_created_at"] < d["ship_date"] + pd.Timedelta(days=1),
                name="order_before_ship",
            ),
            pa.Check(lambda d: d["ingested_at"] >= d["ship_date"], name="ingested_after_ship"),
        ],
        strict=True,
        ordered=False,
    )


SITES = pa.DataFrameSchema(
    {
        "site_id": pa.Column(str, unique=True),
        "country": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{2}$")),
        "capacity_pallets_per_day": pa.Column("int64", pa.Check.gt(0)),
        "operating_days": pa.Column(str),
    },
    strict=True,
)

CUSTOMERS = pa.DataFrameSchema(
    {
        "customer_id": pa.Column(str, unique=True),
        "site_id": pa.Column(str),
        "segment": pa.Column(str),
        "temp_zone": pa.Column(str, pa.Check.isin(["frozen", "chilled"])),
        "onboard_date": pa.Column("datetime64[ns]", nullable=True),
        "churn_date": pa.Column("datetime64[ns]", nullable=True),
    },
    strict=True,
)

PROMO_CALENDAR = pa.DataFrameSchema(
    {
        "promo_id": pa.Column(str, unique=True),
        "customer_id": pa.Column(str),
        "uplift_type": pa.Column(str, pa.Check.isin(["price_cut", "multibuy", "display"])),
        "start_date": pa.Column("datetime64[ns]"),
        "end_date": pa.Column("datetime64[ns]"),
        "announce_date": pa.Column("datetime64[ns]"),
    },
    checks=[
        pa.Check(lambda d: d["end_date"] >= d["start_date"], name="end_after_start"),
        pa.Check(lambda d: d["announce_date"] <= d["start_date"], name="announced_before_start"),
    ],
    strict=True,
)

HOLIDAYS = pa.DataFrameSchema(
    {
        "country": pa.Column(str),
        "date": pa.Column("datetime64[ns]"),
        "name": pa.Column(str),
    },
    unique=["country", "date"],
    strict=True,
)
