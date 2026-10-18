"""Offline vessel matching and review agent.

Imports are lazy so v11 matching can run in Python 3.11 without LangGraph.
The review graph still requires LangGraph.
"""

from importlib import import_module


_EXPORTS = {
    "AgentState": (".schemas", "AgentState"),
    "IntegratedMatchingService": (".integrated_service", "IntegratedMatchingService"),
    "ReviewDecision": (".schemas", "ReviewDecision"),
    "ReviewPolicy": (".review_policy", "ReviewPolicy"),
    "ReviewSignals": (".schemas", "ReviewSignals"),
    "VesselRecord": (".legacy_matching.record_validation", "VesselRecord"),
    "assess_review_need": (".review_policy", "assess_review_need"),
    "build_agent_graph": (".graph", "build_agent_graph"),
    "load_integrated_service": (".integrated_service", "load_integrated_service"),
    "run_agent": (".graph", "run_agent"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(module_name, __name__), attr_name)
