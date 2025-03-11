# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

import json
import datetime
from typing import Any, Callable, Set, Dict, List, Optional


# These are the user-defined functions that can be called by the agent.
def fetch_pump_telemetry(metric: str, statistic: str) -> str:
    """
    Fetch the telemetry data for the Fluid Processing Plant pump.

    :param metric (str): The metric to fetch. Has to be one of the following: 'temperature', 'pressure', 'flowrate'.
    :param statistic (str): The statistic to fetch. Has to be one of the following: 'mean', 'min', 'max'.
    :return: The telemetry data for the pump in JSON format.
    :rtype: str
    """
    value = None
    unit = None

    match metric:
        case "temperature":
            unit = "Celsius"
            match statistic:
                case "mean":
                    value = 75.0
                case "min":
                    value = 60.0
                case "max":
                    value = 90.0
                case _:
                    return json.dumps(
                        {
                            "message": f"Invalid statistic: {statistic}. Please provide a valid statistic: 'mean', 'min', 'max'."
                        }
                    )
        case "pressure":
            unit = "Bar"
            match statistic:
                case "mean":
                    value = 5.0
                case "min":
                    value = 3.0
                case "max":
                    value = 7.0
                case _:
                    return json.dumps(
                        {
                            "message": f"Invalid statistic: {statistic}. Please provide a valid statistic: 'mean', 'min', 'max'."
                        }
                    )
        case "flowrate":
            unit = "L/min"
            match statistic:
                case "mean":
                    value = 100.0
                case "min":
                    value = 80.0
                case "max":
                    value = 120.0
                case _:
                    return json.dumps(
                        {
                            "message": f"Invalid statistic: {statistic}. Please provide a valid statistic: 'mean', 'min', 'max'."
                        }
                    )
        case _:
            return json.dumps(
                {
                    "message": f"Invalid metric: {metric}. Please provide a valid metric: 'temperature', 'pressure', 'flowrate'."
                }
            )

    return json.dumps(
        {"metric": metric, "statistic": statistic, "value": value, "unit": unit}
    )


def fetch_current_datetime(format: Optional[str] = None) -> str:
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


def think_tool(thoughts: str) -> str:
    """
    This tool is for you to think about the human's question and the context in which it was asked.
    Use this tool to reason over the question, the context, and aspects of the problem that you are
    considering and plan required actions. Your thoughts will be recorded and provided back to you to
    help you formulate the final response.

    :param thoughts (str): Use this field as the scratchpad to put down your stream of consciousness as you mull the original question.
    :return: Reflects the toughts back to you.
    :rtype: str
    """
    return json.dumps(
        {"message": f"Here are your internal thoughts about the question: {thoughts}"}
    )


# Statically defined user functions for fast reference
user_functions: Set[Callable[..., Any]] = {think_tool, fetch_current_datetime, fetch_pump_telemetry}
