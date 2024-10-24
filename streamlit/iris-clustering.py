import streamlit as st
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import plotly.express as px


def main():
    st.title("Iris Clustering")

    # preparing UI for hyper-parameters
    with st.sidebar:
        algo = st.selectbox("Method", ["KMeans", "DBSCAN"])

        if algo == "KMeans":
            k = st.slider("K", min_value=2, max_value=10, step=1)
        elif algo == "DBSCAN":
            eps = st.number_input("eps", min_value=0.1, max_value=5.0, step=0.05)
            samples = st.slider("Min. Samples", min_value=1, max_value=10, step=1)

    # preparing dataset
    df = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names)
    
    # clustering algorithms
    if algo == "KMeans":
        model = KMeans(n_clusters=k).fit(df)
        df["clusters"] = model.predict(df)
        # st.write(clusters)
    elif algo == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=samples)
        df["clusters"] = model.fit_predict(df)

    # visulization
    columns = st.multiselect("Columns", options=load_iris().feature_names, max_selections=3,
                            default=['sepal length (cm)','sepal width (cm)','petal length (cm)'])
    fig = px.scatter_3d(df,
                        x=columns[0],
                        y=columns[1],
                        z=columns[2],
                        color='clusters',
                        height=500)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
    