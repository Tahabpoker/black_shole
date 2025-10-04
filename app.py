import streamlit as st

st.title("Black-Scholes Pricing Model")
st.table({
    "Current Asset Price": [99.93],
    "Strike Price": [70.0],
    "Time to Maturity": [1.0],
    "Volatility": [0.20],
    "Risk-Free Interest Rate": [0.05]
})

st.markdown(
    """
    <style>
    .box {
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
    }
    .call-box {
        background-color: #28a745;
    }
    .put-box {
        background-color: #dc3545;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="box call-box">
            <div >CALL Value</div>
            <div >$10.47</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="box put-box">
            <div >PUT Value</div>
            <div >$5.55</div>
        </div>
        """,
        unsafe_allow_html=True
    )