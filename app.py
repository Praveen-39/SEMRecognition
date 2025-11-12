"""
Entrypoint for the Streamlit app.
This small wrapper lets hosting platforms run `streamlit run app.py` while keeping the main app in `results/ph5_streamlit.py`.
"""

from results.ph5_streamlit import main

if __name__ == "__main__":
    main()
