import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pdfplumber
import json
import io

st.set_page_config(
    page_title="Data Analysis Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Analysis Tool")
st.markdown("Upload any data file and explore, visualize, and analyze your data instantly.")

SUPPORTED_TYPES = ["csv", "xlsx", "xls", "pdf", "json", "tsv", "txt"]


def load_csv(file):
    try:
        return pd.read_csv(file)
    except Exception:
        return pd.read_csv(file, encoding="latin1")


def load_excel(file):
    xl = pd.ExcelFile(file)
    if len(xl.sheet_names) > 1:
        sheet = st.selectbox("Select a sheet", xl.sheet_names)
    else:
        sheet = xl.sheet_names[0]
    return pd.read_excel(file, sheet_name=sheet)


def load_pdf(file):
    tables = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_tables()
            for table in extracted:
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
    if tables:
        if len(tables) > 1:
            idx = st.selectbox("Multiple tables found. Select one:", range(len(tables)),
                               format_func=lambda i: f"Table {i + 1}")
            return tables[idx]
        return tables[0]
    st.warning("No tables found in the PDF. Only PDFs with tables can be analyzed.")
    return None


def load_json(file):
    data = json.load(file)
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.DataFrame([data])
    return None


def load_tsv(file):
    try:
        return pd.read_csv(file, sep="\t")
    except Exception:
        return pd.read_csv(file, sep="\t", encoding="latin1")


def load_file(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "csv":
        return load_csv(uploaded_file)
    elif ext in ["xlsx", "xls"]:
        return load_excel(uploaded_file)
    elif ext == "pdf":
        return load_pdf(uploaded_file)
    elif ext == "json":
        return load_json(uploaded_file)
    elif ext in ["tsv", "txt"]:
        return load_tsv(uploaded_file)
    else:
        st.error(f"File type '.{ext}' is not supported yet.")
        return None


def generate_ai_summary(df):
    insights = []
    rows, cols = df.shape
    insights.append(f"Your dataset has **{rows:,} rows** and **{cols} columns**.")

    missing = df.isnull().sum().sum()
    if missing > 0:
        pct = round((missing / (rows * cols)) * 100, 1)
        insights.append(f"There are **{missing:,} missing values** ({pct}% of your data). You may want to review or clean these.")
    else:
        insights.append("Your data has **no missing values** — great data quality!")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()

    if numeric_cols:
        insights.append(f"Found **{len(numeric_cols)} numeric column(s)**: {', '.join(numeric_cols)}.")

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            mean_val = series.mean()
            std_val = series.std()
            min_val = series.min()
            max_val = series.max()
            skew = series.skew()

            insights.append(
                f"**{col}** ranges from {round(min_val, 2)} to {round(max_val, 2)}, "
                f"with an average of {round(mean_val, 2)}."
            )

            if abs(skew) > 1:
                direction = "right (positively)" if skew > 0 else "left (negatively)"
                insights.append(
                    f"**{col}** is skewed {direction}, meaning most values are concentrated on one side."
                )

            outlier_threshold = 3
            z_scores = np.abs((series - mean_val) / std_val) if std_val > 0 else pd.Series([0] * len(series))
            outliers = (z_scores > outlier_threshold).sum()
            if outliers > 0:
                insights.append(
                    f"**{col}** has **{outliers} potential outlier(s)** — values that are unusually high or low compared to the rest."
                )

        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            strong_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) >= 0.7:
                        direction = "positively" if val > 0 else "negatively"
                        strong_pairs.append(
                            f"**{numeric_cols[i]}** and **{numeric_cols[j]}** are strongly {direction} correlated ({round(val, 2)})"
                        )
            if strong_pairs:
                insights.append("Strong relationships found: " + "; ".join(strong_pairs) + ".")
            else:
                insights.append("No strong correlations found between numeric columns.")

    if text_cols:
        insights.append(f"Found **{len(text_cols)} text column(s)**: {', '.join(text_cols)}.")
        for col in text_cols:
            n_unique = df[col].nunique()
            top_val = df[col].value_counts().idxmax() if not df[col].dropna().empty else "N/A"
            top_count = df[col].value_counts().max() if not df[col].dropna().empty else 0
            insights.append(
                f"**{col}** has {n_unique} unique values. The most common is **'{top_val}'** "
                f"(appears {top_count} times)."
            )

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        insights.append(f"Found **{duplicates} duplicate row(s)** in your data. Consider removing them for cleaner analysis.")
    else:
        insights.append("No duplicate rows found.")

    return insights


def recommend_charts(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    recommendations = []

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                val = abs(corr.iloc[i, j])
                if val >= 0.5:
                    recommendations.append({
                        "chart": "Scatter Plot",
                        "reason": f"**{numeric_cols[i]}** and **{numeric_cols[j]}** are related (correlation: {round(val, 2)}). A scatter plot will show this relationship clearly.",
                        "x": numeric_cols[i],
                        "y": numeric_cols[j],
                        "type": "scatter"
                    })

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        skew = abs(series.skew())
        if skew > 0.5:
            recommendations.append({
                "chart": "Histogram",
                "reason": f"**{col}** has an uneven distribution (skew: {round(skew, 2)}). A histogram will show how values are spread out.",
                "col": col,
                "type": "histogram"
            })

    for col in text_cols:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 15:
            if numeric_cols:
                recommendations.append({
                    "chart": "Bar Chart",
                    "reason": f"**{col}** has {n_unique} categories. A bar chart is perfect for comparing values across these groups.",
                    "x": col,
                    "y": numeric_cols[0],
                    "type": "bar"
                })
            recommendations.append({
                "chart": "Pie Chart",
                "reason": f"**{col}** has {n_unique} categories. A pie chart will show the share of each category.",
                "col": col,
                "type": "pie"
            })

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        std_val = series.std()
        mean_val = series.mean()
        cv = (std_val / mean_val) if mean_val != 0 else 0
        if abs(cv) > 0.5:
            recommendations.append({
                "chart": "Box Plot",
                "reason": f"**{col}** has high variability. A box plot will reveal the spread, median, and any outliers.",
                "col": col,
                "type": "box"
            })

    if not recommendations and numeric_cols:
        recommendations.append({
            "chart": "Bar Chart",
            "reason": f"A simple bar chart works well to compare **{numeric_cols[0]}** across your data.",
            "x": df.columns[0],
            "y": numeric_cols[0],
            "type": "bar"
        })

    return recommendations[:5]


uploaded_file = st.file_uploader(
    "Upload your file here",
    type=SUPPORTED_TYPES,
    help="Supported formats: CSV, Excel (.xlsx/.xls), PDF, JSON, TSV, TXT"
)

if uploaded_file is not None:
    with st.spinner("Loading your data..."):
        df = load_file(uploaded_file)

    if df is not None:
        df.columns = [str(c).strip() for c in df.columns]
        df = df.replace("", np.nan)

        st.success(f"File loaded successfully — {df.shape[0]:,} rows and {df.shape[1]} columns")

        tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🤖 AI Summary", "📋 Data Preview", "📈 Statistics", "📊 Charts & Recommendations", "🔍 Filter & Search", "⬇️ Download"
        ])

        with tab0:
            st.subheader("🤖 AI-Powered Data Summary")
            st.markdown("Here is what the AI found in your data:")
            with st.spinner("Analyzing your data..."):
                insights = generate_ai_summary(df)
            for i, insight in enumerate(insights):
                st.markdown(f"- {insight}")

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) >= 2:
                st.markdown("---")
                st.subheader("Correlation Heatmap")
                corr = df[numeric_cols].corr().round(2)
                fig = px.imshow(corr, text_auto=True, color_continuous_scale="Blues",
                                title="How strongly columns relate to each other")
                st.plotly_chart(fig, use_container_width=True)

        with tab1:
            st.subheader("Data Preview")
            rows = st.slider("Rows to display", 5, min(500, len(df)), min(20, len(df)))
            st.dataframe(df.head(rows), use_container_width=True)
            st.caption(f"Showing {rows} of {len(df):,} total rows")

            st.subheader("Column Info")
            info = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Non-Null Count": df.count().values,
                "Null Count": df.isnull().sum().values,
                "Unique Values": df.nunique().values
            })
            st.dataframe(info, use_container_width=True)

        with tab2:
            st.subheader("Summary Statistics")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

                st.subheader("Column Highlights")
                chosen = st.selectbox("Pick a numeric column", numeric_cols)
                col1, col2, col3 = st.columns(3)
                col1.metric("Min", round(df[chosen].min(), 2))
                col2.metric("Max", round(df[chosen].max(), 2))
                col3.metric("Average", round(df[chosen].mean(), 2))
            else:
                st.info("No numeric columns found for statistics.")

            text_cols = df.select_dtypes(include="object").columns.tolist()
            if text_cols:
                st.subheader("Text Column Summaries")
                chosen_text = st.selectbox("Pick a text column", text_cols)
                counts = df[chosen_text].value_counts().reset_index()
                counts.columns = [chosen_text, "Count"]
                st.dataframe(counts, use_container_width=True)

        with tab3:
            st.subheader("📊 Charts & Visualizations")

            st.markdown("### 💡 Recommended Charts for Your Data")
            recs = recommend_charts(df)
            if recs:
                for idx, rec in enumerate(recs):
                    with st.expander(f"Recommendation {idx + 1}: {rec['chart']} — click to view"):
                        st.markdown(rec["reason"])
                        try:
                            if rec["type"] == "scatter":
                                fig = px.scatter(df, x=rec["x"], y=rec["y"],
                                                 title=f"{rec['x']} vs {rec['y']}")
                                st.plotly_chart(fig, use_container_width=True)
                            elif rec["type"] == "histogram":
                                fig = px.histogram(df, x=rec["col"], nbins=30,
                                                   title=f"Distribution of {rec['col']}")
                                st.plotly_chart(fig, use_container_width=True)
                            elif rec["type"] == "bar":
                                fig = px.bar(df, x=rec["x"], y=rec["y"],
                                             title=f"{rec['y']} by {rec['x']}")
                                st.plotly_chart(fig, use_container_width=True)
                            elif rec["type"] == "pie":
                                counts = df[rec["col"]].value_counts().reset_index()
                                counts.columns = [rec["col"], "Count"]
                                fig = px.pie(counts, names=rec["col"], values="Count",
                                             title=f"Share of {rec['col']}")
                                st.plotly_chart(fig, use_container_width=True)
                            elif rec["type"] == "box":
                                fig = px.box(df, y=rec["col"],
                                             title=f"Spread of {rec['col']}")
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            st.info("Could not render this chart with your data.")
            else:
                st.info("No specific chart recommendations for this data. Use the manual section below.")

            st.markdown("---")
            st.markdown("### 🛠️ Build Your Own Chart")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            all_cols = df.columns.tolist()

            chart_type = st.selectbox("Chart type", [
                "Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"
            ])

            if chart_type == "Bar Chart":
                x = st.selectbox("X axis (categories)", all_cols)
                y = st.selectbox("Y axis (values)", numeric_cols) if numeric_cols else None
                if y:
                    fig = px.bar(df, x=x, y=y, title=f"{y} by {x}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Line Chart":
                x = st.selectbox("X axis", all_cols)
                y = st.multiselect("Y axis (values)", numeric_cols, default=numeric_cols[:1] if numeric_cols else [])
                if y:
                    fig = px.line(df, x=x, y=y, title="Line Chart")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Scatter Plot":
                if len(numeric_cols) >= 2:
                    x = st.selectbox("X axis", numeric_cols, index=0)
                    y = st.selectbox("Y axis", numeric_cols, index=1)
                    color = st.selectbox("Color by (optional)", ["None"] + all_cols)
                    color = None if color == "None" else color
                    fig = px.scatter(df, x=x, y=y, color=color, title=f"{x} vs {y}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns for a scatter plot.")

            elif chart_type == "Pie Chart":
                label_col = st.selectbox("Label column", all_cols)
                value_col = st.selectbox("Value column", numeric_cols) if numeric_cols else None
                if value_col:
                    fig = px.pie(df, names=label_col, values=value_col, title=f"{value_col} by {label_col}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Histogram":
                col = st.selectbox("Column", numeric_cols) if numeric_cols else None
                if col:
                    bins = st.slider("Number of bins", 5, 100, 20)
                    fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Box Plot":
                y = st.selectbox("Value column", numeric_cols) if numeric_cols else None
                x = st.selectbox("Group by (optional)", ["None"] + all_cols)
                x = None if x == "None" else x
                if y:
                    fig = px.box(df, x=x, y=y, title=f"Box Plot of {y}")
                    st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Filter & Search Your Data")
            search_col = st.selectbox("Search in column", df.columns.tolist())
            search_val = st.text_input("Search for a value")
            filtered = df[df[search_col].astype(str).str.contains(search_val, case=False, na=False)] if search_val else df

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                st.markdown("**Filter by numeric range:**")
                range_col = st.selectbox("Select numeric column to filter", numeric_cols)
                col_min = float(df[range_col].min())
                col_max = float(df[range_col].max())
                if col_min < col_max:
                    range_vals = st.slider("Range", col_min, col_max, (col_min, col_max))
                    filtered = filtered[
                        (filtered[range_col] >= range_vals[0]) &
                        (filtered[range_col] <= range_vals[1])
                    ]

            st.write(f"Showing **{len(filtered):,}** matching rows")
            st.dataframe(filtered, use_container_width=True)

        with tab5:
            st.subheader("Download Your Data")
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv_data,
                file_name="analyzed_data.csv",
                mime="text/csv"
            )

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            st.download_button(
                label="⬇️ Download as Excel",
                data=excel_buffer.getvalue(),
                file_name="analyzed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("👆 Upload a file above to get started. Supported formats: CSV, Excel, PDF, JSON, TSV, TXT")
    st.markdown("""
    ### What this tool can do:
    - **🤖 AI Summary** — auto-detects patterns, outliers, correlations and missing data
    - **Preview** your data in a clean table
    - **Analyze** statistics like min, max, average, and value counts
    - **Visualize** with bar charts, line charts, scatter plots, pie charts, histograms, and box plots
    - **Filter & Search** through your data easily
    - **Download** the results as CSV or Excel
    """)
