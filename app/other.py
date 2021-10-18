import pandas as pd
import streamlit as st


def other():
    st.text('This predictions are based on')

    st.latex(r'''
    a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
    \sum_{k=0}^{n-1} ar^k =
    a \left(\frac{1-r^{n}}{1-r}\right)
    ''')

    st.write('<h1>Hello Title</h1>', unsafe_allow_html=True)

    st.dataframe(pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40],
    }))

    st.code('''
    def hello_streamlit():
        return 10!
    ''', language='python')

    st.json({
        'foo': 'bar',
        'baz': 'boz',
        'stuff': [
            'stuff 1',
            'stuff 2',
            'stuff 3',
            'stuff 5',
        ],
    })

    st.graphviz_chart('''
        digraph {
            run -> intr
            intr -> runbl
            runbl -> run
            run -> kernel
            kernel -> zombie
            kernel -> sleep
            kernel -> runmem
            sleep -> swap
            swap -> runswap
            runswap -> new
            runswap -> runmem
            new -> runmem
            sleep -> runmem
        }
    ''')
