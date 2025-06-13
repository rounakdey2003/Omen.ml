import numpy as np
import plotly.graph_objects as go
import streamlit as st

with st.spinner('Loading...'):
    st.toast('Checking Environment')


    def main():
        titleCol1, titleCol2, titleCol3 = st.columns([1, 1, 1])
        with titleCol1:
            pass
        with titleCol2:
            st.title('***Math GPT*** 📚')
            st.write(':grey[Solve basic-complex graphical problems]')
        with titleCol3:
            pass

        st.divider()

        plot = False
        tab1, tab2, tab3 = st.tabs(["Calculator", "Linear Equation", "Graph Plotter"])

        with tab1:
            with st.container(height=400, border=True):
                st.header(":grey[Calculator]")
                input = st.text_input("Enter a valid calculation:")
                st.warning("Rule =  2*3+5")
                if st.button("Calculate"):
                    result = eval(input)
                    st.success(f"**Result:** {result}")

        with tab2:
            with st.container(height=800, border=True):
                st.header(":grey[Linear Equation]")
                st.info(
                    "2x -3y = 5 :{coefficient of x is :grey[2], coefficient of y is :grey[-3], coefficient of c is :grey[5]}")
                a1 = st.number_input("Enter coefficient 'a' for Equation 1:")
                b1 = st.number_input("Enter coefficient 'b' for Equation 1:")
                c1 = st.number_input("Enter coefficient 'c' for Equation 1:")
                a2 = st.number_input("Enter coefficient 'a' for Equation 2:")
                b2 = st.number_input("Enter coefficient 'b' for Equation 2:")
                c2 = st.number_input("Enter coefficient 'c' for Equation 2:")


                eq1 = [a1, b1, c1]
                eq2 = [a2, b2, c2]
                a1, b1, c1 = eq1
                a2, b2, c2 = eq2

                A = np.array([[a1, b1], [a2, b2]])
                B = np.array([c1, c2])
                try:
                    solution = np.linalg.solve(A, B)
                    st.success(f":green[Result]: x = {solution[0]}, y = {solution[1]}")

                except np.linalg.LinAlgError:
                    st.warning("No unique solution exists for the given equations.")

        with tab3:
            with st.container(height=700, border=True):
                st.header(":grey[Graph Plotter]")

                a1 = st.number_input('Coefficient (a1) of first equation', value=1.0)
                b1 = st.number_input('Coefficient (b1) of first equation', value=0.0)
                a2 = st.number_input('Coefficient (a2) of second equation', value=1.0)
                b2 = st.number_input('Coefficient (b2) of second equation', value=0.0)

                st.write('You entered the following equations:')
                st.info(f'Equation 1: y = {a1}x + {b1}')
                st.info(f'Equation 2: y = {a2}x + {b2}')

                x = np.linspace(-10, 10, 400)
                y1 = a1 * x + b1
                y2 = a2 * x + b2

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name=f'y = {a1}x + {b1}'))
                fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name=f'y = {a2}x + {b2}'))

                fig.update_layout(
                    title='Graphical Representation of Linear Equations',
                    xaxis_title='x',
                    yaxis_title='y',
                    showlegend=True,
                    shapes=[
                        dict(
                            type="line",
                            x0=min(x),
                            y0=0,
                            x1=max(x),
                            y1=0,
                            line=dict(
                                color="black",
                                width=0.5
                            )
                        ),
                        dict(
                            type="line",
                            x0=0,
                            y0=min(y1.min(), y2.min()),
                            x1=0,
                            y1=max(y1.max(), y2.max()),
                            line=dict(
                                color="black",
                                width=0.5
                            )
                        )
                    ],
                    xaxis=dict(
                        gridcolor='gray',
                        gridwidth=0.5,
                        zeroline=False
                    ),
                    yaxis=dict(
                        gridcolor='gray',
                        gridwidth=0.5,
                        zeroline=False
                    )
                )

            with st.expander("See Graph", expanded=True):
                st.plotly_chart(fig)


    if __name__ == "__main__":
        main()

    st.toast(':green[Ready!]')
