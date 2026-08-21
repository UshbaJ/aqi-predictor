"""
check_feature_group.py

Quick sanity check: connects to Hopsworks and reads back the row count
of aqi_weather_features to confirm the upload actually succeeded,
without needing to navigate the UI.

Usage:
    python src/check_feature_group.py
"""

import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()


def main():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name="aqi_weather_features", version=1)

    # commit_details() queries a metadata REST endpoint directly - it does
    # NOT go through the ArrowFlight/DuckDB read service that's currently
    # hitting a server-side bug ("Set changed size during iteration").
    # This tells us definitively how many rows were committed without
    # needing to stream the actual data back.
    commits = fg.commit_details()
    print(f"\nNumber of commits: {len(commits)}")
    for commit_time, details in commits.items():
        print(f"\nCommit at {commit_time}:")
        print(f"  Rows inserted: {details.get('rowsInserted')}")
        print(f"  Rows updated:  {details.get('rowsUpdated')}")
        print(f"  Rows deleted:  {details.get('rowsDeleted')}")


if __name__ == "__main__":
    main()
