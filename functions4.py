# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
import pandas as pd
import numpy as np
import json
import datetime
from typing import Any, Callable, Set, Dict, List, Optional

class Functions4:
    """ These are the user-defined functions that can be called by the agent."""
    def __init__(self):
        self.df = pd.DataFrame(
            {
                "temperature": np.random.normal(loc=80, scale=5, size=21),
                "flow_rate": np.random.normal(loc=15, scale=5, size=21),
            },
            index=pd.date_range(start="2023-01-01 00:00:00", periods=21, freq="15min"),
        )

        self.user_functions = {
            self.think_tool,
            self.fetch_current_datetime,
            self.fetch_pump_telemetry
        }

    def fetch_pump_telemetry(self, measure: str, statistic: str) -> str:
        """
        Get telemetry data statistic for a physical measure on the pump.

        :param measure (str): The metric to fetch. Must be either 'temperature' or 'flow_rate'.
        :param statistic (str): The statistic to fetch. Must be a valid pandas time-series aggregation function.
        :return: The telemetry data for the pump in JSON format.
        :rtype: str
        """
        value = None
        unit = None

        # Check that measure is a valid column in the DataFrame
        if measure not in self.df.columns:
            raise ValueError(f"Invalid measure: {measure}. Must be one of {self.df.columns}")
        # Check that statistic is a valid pandas aggregation function
        valid_statistics = [
            "mean",
            "median",
            "min",
            "max",
            "sum",
            "std",
            "var",
            "sem",
            "skew",
            "kurt",
            "prod",
        ]
        if statistic not in valid_statistics:
            raise ValueError(
                f"Invalid statistic: {statistic}. Must be one of {valid_statistics}"
            )
        # Set the unit based on the measure
        match measure:
            case "temperature":
                unit = "Fahrenheit"
            case "flow_rate":
                unit = "Gallons/min"

        # Return the requested statistic for the specified measure
        value = self.df[measure].agg(statistic)
        return json.dumps(
            {"measure": measure, "statistic": statistic, "value": value, "unit": unit}
        )


    def fetch_current_datetime(self, format: Optional[str] = None) -> str:
        """
        Get the current time as a JSON string, optionally formatted.

        :param format (Optional[str]): The format in which to return the current time. Defaults to None, which uses a standard format.
        :return: The current time in JSON format.
        :rtype: str
        """
        current_time = datetime.datetime.now()

        # Use the provided format if available, else use a default format
        if format:
            time_format = format
        else:
            time_format = "%Y-%m-%d %H:%M:%S"

        time_json = json.dumps({"current_time": current_time.strftime(time_format)})
        return time_json


    def think_tool(self, thoughts: str) -> str:
        """
        This tool is for you to think about the human's question and the context in which it was asked.
        Use this tool to reason over the question, the context, and aspects of the problem that you are
        considering and plan required actions. Your thoughts will be recorded and provided back to you to
        help you formulate the final response.

        :param thoughts (str): Use this field as the scratchpad to put down your stream of consciousness as you mull the original question.
        :return: Reflects the toughts back to you.
        :rtype: str
        """
        return f"Here are your internal thoughts about the question: {thoughts}"
