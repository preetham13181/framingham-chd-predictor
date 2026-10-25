import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Framingham data explorer", page_icon="🫀", layout="wide")

st.title("Framingham data explorer")
st.caption("Upload a CSV to explore its numeric columns and try a simple regression.")
st.info(
    "This app demonstrates a two-input linear regression to estimate a selected "
    "numeric outcome; if the estimate is wrong, it can mislead decisions, so this "
    "Framingham example must not be used to guide medical care."
)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Choose a CSV file to begin. Your file is used in this app session and is not saved by this code.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"I couldn't read that CSV. Check that it is a valid, comma-separated file. Details: {exc}")
    st.stop()

if df.empty:
    st.error("This CSV has no data rows. Upload a file with a header and data rows.")
    st.stop()

df.columns = [str(column).strip() for column in df.columns]
if df.columns.duplicated().any():
    st.error("Some column names are duplicated after removing extra spaces. Rename them and upload the CSV again.")
    st.stop()

st.subheader("File preview")
st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
st.dataframe(df.head(10), use_container_width=True)

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_columns) < 3:
    st.error("The app needs at least three numeric columns: two inputs and one regression target.")
    st.stop()

# Explore one numeric column. A scatter plot needs a second column for its x-axis.
st.subheader("Explore a column")
plot_column = st.selectbox("Choose the column to chart", numeric_columns)
chart_type = st.selectbox("Choose a chart type", ["Line", "Scatter", "Histogram"])

if chart_type == "Scatter":
    scatter_options = [column for column in numeric_columns if column != plot_column]
    x_column = st.selectbox("Choose the x-axis column", scatter_options)
    chart_data = df[[x_column, plot_column]].apply(pd.to_numeric, errors="coerce").dropna()

    if chart_data.empty:
        st.warning("These columns don't have any complete numeric row pairs to plot.")
    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.scatter(chart_data[x_column], chart_data[plot_column], alpha=0.55, color="#3157A4")
        ax.set_xlabel(x_column)
        ax.set_ylabel(plot_column)
        ax.set_title(f"{plot_column} versus {x_column}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

elif chart_type == "Line":
    values = pd.to_numeric(df[plot_column], errors="coerce").dropna()
    if values.empty:
        st.warning("This column doesn't contain values that can be plotted.")
    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(values.index + 1, values.values, color="#3157A4", linewidth=1.2)
        ax.set_xlabel("Row number in the uploaded file")
        ax.set_ylabel(plot_column)
        ax.set_title(f"{plot_column} by row order")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

else:  # Histogram
    values = pd.to_numeric(df[plot_column], errors="coerce").dropna()
    if values.empty:
        st.warning("This column doesn't contain values that can be plotted.")
    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(values, bins=25, color="#3157A4", edgecolor="white")
        ax.set_xlabel(plot_column)
        ax.set_ylabel("Number of rows")
        ax.set_title(f"Distribution of {plot_column}")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# Choose two predictors and a numeric target for linear regression.
st.subheader("Try a two-input linear regression")
st.caption("The app drops rows missing any of the three selected values, then evaluates on a held-out test split.")

def default_index(options, preferred):
    return options.index(preferred) if preferred in options else 0

preferred_target = "TenYearCHD" if "TenYearCHD" in numeric_columns else numeric_columns[-1]
target_column = st.selectbox(
    "Target: what should the model estimate?",
    numeric_columns,
    index=default_index(numeric_columns, preferred_target),
)

feature_options = [column for column in numeric_columns if column != target_column]
if len(feature_options) < 2:
    st.error("Choose a CSV with two numeric input columns in addition to the target.")
    st.stop()

preferred_first = "Age" if "Age" in feature_options else feature_options[0]
feature_one = st.selectbox(
    "Input 1",
    feature_options,
    index=default_index(feature_options, preferred_first),
)
second_options = [column for column in feature_options if column != feature_one]
preferred_second = next(
    (column for column in ["SysBP", "BMI", "TotChol", "Glucose"] if column in second_options),
    second_options[0],
)
feature_two = st.selectbox(
    "Input 2",
    second_options,
    index=default_index(second_options, preferred_second),
)

model_data = df[[feature_one, feature_two, target_column]].copy()
for column in [feature_one, feature_two, target_column]:
    model_data[column] = pd.to_numeric(model_data[column], errors="coerce")
model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna()

st.caption(f"Rows available to this model after removing missing or invalid values: {len(model_data):,}")
if len(model_data) < 10:
    st.error("At least 10 complete rows are needed for this small train/test demonstration. Try another target or input pair.")
    st.stop()

# Keep the same feature names and order in training and prediction.
X = model_data[[feature_one, feature_two]]
y = model_data[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)

metric_one, metric_two = st.columns(2)
metric_one.metric("Test-set mean absolute error", f"{mean_absolute_error(y_test, test_predictions):.3f}")
metric_two.metric("Test-set R²", f"{r2_score(y_test, test_predictions):.3f}")
st.caption("These are teaching metrics from one random split, not evidence that the model is suitable for real decisions.")

if target_column == "TenYearCHD":
    st.warning(
        "For the 0/1 TenYearCHD target, linear regression returns an unbounded numeric estimate. "
        "It is not a validated risk score or probability and must not be used for health decisions."
    )

st.markdown("#### Enter values to get a model estimate")
st.caption("Values outside the ranges seen in the uploaded file are extrapolations and may be especially unreliable.")

with st.form("prediction_form"):
    input_left, input_right = st.columns(2)
    value_one = input_left.number_input(
        feature_one,
        value=float(model_data[feature_one].median()),
        step=1.0,
        format="%.2f",
        key=f"prediction_{feature_one}",
    )
    value_two = input_right.number_input(
        feature_two,
        value=float(model_data[feature_two].median()),
        step=1.0,
        format="%.2f",
        key=f"prediction_{feature_two}",
    )
    submitted = st.form_submit_button("Estimate target")

if submitted:
    entered_values = {feature_one: value_one, feature_two: value_two}
    outside_range = []
    for column, value in entered_values.items():
        observed_min = float(model_data[column].min())
        observed_max = float(model_data[column].max())
        if value < observed_min or value > observed_max:
            outside_range.append(column)

    if outside_range:
        st.warning("Outside the observed data range for: " + ", ".join(outside_range) + ". Treat this estimate cautiously.")

    # The DataFrame column names and order exactly match X used for model.fit().
    one_case = pd.DataFrame([[value_one, value_two]], columns=[feature_one, feature_two])
    estimate = float(model.predict(one_case)[0])
    st.metric(f"Estimated {target_column}", f"{estimate:.3f}")