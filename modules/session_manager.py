import streamlit as st


DEFAULT_STATE = {
    "dataset": None,
    "processed_data": None,
    "results": None,
    "comparison_results": [],
    "trained_models": [],
    "current_model": None
}


def initialize_session():

    for key, value in DEFAULT_STATE.items():

        if key not in st.session_state:

            st.session_state[key] = value