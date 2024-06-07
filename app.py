import streamlit as st
import pickle
import numpy as np
import sklearn
from streamlit_option_menu import option_menu
from PIL import Image

# Functions
def predict_status(cust, itmtp, aplcn, tkns, wth, prdrf, selpr, dtdf):
    with open("C:/Users/DELL-22/Desktop/COPPER MODELING/Classification_Model.pkl", "rb") as f:
        model_class = pickle.load(f)

    user_data = np.array([[cust, itmtp, aplcn, tkns, wth, prdrf, selpr, dtdf]])
    y_pred = model_class.predict(user_data)

    return y_pred[0]

def predict_selling_price(cust, stt, itmtp, aplcn, tkns, wth, prdrf, dtdf):
    with open("C:/Users/DELL-22/Desktop/COPPER MODELING/Regression_Model.pkl", "rb") as f:
        model_regg = pickle.load(f)

    user_data = np.array([[cust, stt, itmtp, aplcn, tkns, wth, prdrf, dtdf]])
    y_pred = model_regg.predict(user_data)
    ac_y_pred = np.exp(y_pred[0])

    return ac_y_pred

st.set_page_config(layout="wide")
st.title(":violet[Industrial Copper Modeling Prediction Using ML Algorithm] | by Thigazhvan")


option = option_menu('MENU', options=["About the Poject","Predict the sellling price", "Predict the status"])

if option == "About the Poject":
    st.title(":blue[Project Industrial copper modeling]")
    st.header(":blue[Domain]")
    st.markdown("Manufacturing Industry")
    st.header(":blue[Skills take away From This Project]")
    st.markdown(" Python scripting, Data Preprocessing,EDA, Streamlit,Github")
    col1,col2,col3=st.columns(3)
    with col1:
        st.image("C:/Users/DELL-22/Desktop/COPPER MODELING/python.png")
    with col2:
        st.image("C:/Users/DELL-22/Desktop/COPPER MODELING/download.png")
    with col3:
        st.image("C:/Users/DELL-22/Desktop/COPPER MODELING/streamlit.png")

    st.header("Data Understanding:")
    st.markdown("Identify the types of variables (continuous, categorical) and their distributions. Some rubbish values are present in ‘Material_Reference’ which starts with ‘00000’ value which should be converted into null. Treat reference columns as categorical variables. INDEX may not be useful.")
    st.header("Data Preprocessing:")
    st.markdown(" Handle missing values with mean/median/mode.Treat Outliers using IQR or Isolation Forest from sklearn library.Identify Skewness in the dataset and treat skewness with appropriate data transformations, such as log transformation(which is best suited to transform target variable-train, predict and then reverse transform it back to original scale eg:dollars), boxcox transformation, or other techniques, to handle high skewness in continuous variables.Encode catgorical variables using suitable techniques, such as one-hot encoding, label encoding, or ordinal encoding, based on their nature and relationship with the target variable.")
    st.header("EDA:")
    st.markdown("Try visualizing outliers and skewness(before and after treating skewness) using Seaborn’s boxplot, distplot, violinplot.")
    st.header("Feature Engineering:")
    st.markdown("Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data. And drop highly correlated columns using SNS HEATMAP.")
    st.header("Model Building and Evaluation::")
    st.markdown("Split the dataset into training and testing/validation sets. Train and evaluate different classification models, such as ExtraTreesClassifier, XGBClassifier, or Logistic Regression, using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve.")


if option == "Predict the status":
    st.header("Prediction (Won / Lose)")
    st.header("Prediction Based on RandonForest Classification Algorithm")
    st.write("Enter the vslues below the tabs based on columns to predict the status")

    
    customer = st.number_input(label="Customer(Min:17.21910, Max:17.23015)", format="%0.15f")
    thickness = st.number_input(label="Thickness (Min:-1.71479, Max:3.28154)", format="%0.15f")
    item_type = st.number_input(label="Item type (Min:1.0, Max:7.0)")
    application = st.number_input(label="Application(Min:2.0, Max:87.5)")
    width = st.number_input(label="Width(Min:700.0, Max:1980.0)")
    product_ref = st.number_input(label="Product_ref(Min:611728, Max:1722207579)")
    selling_price = st.number_input(label="Selling Price(Min:5.97503, Max:7.39036)", format="%0.15f")
    date_differ = st.number_input(label="Datedifference(Min:0, Max:689)")
    st.write("Min,Max values were given for Refference")
    button = st.button(":green[Predict the status]", use_container_width=True)

    if button:
        status = predict_status(customer, item_type, application, thickness, width, product_ref, selling_price, date_differ)
        if status == 1:
            st.write("## :green[The Status is WON]")
        else:
            st.write("## :red[The Status is LOSE]")

    st.write("The status likely describes the current status of the transaction or item. This information can be used to track the progress of orders or transactions, such as 'lose' or 'Won.'")

if option == "Predict the sellling price":
    st.header("**Prediction of selling price**")
    st.header("Prediction is based on the RandomForest Regression Algorithm")
    st.write(" ")


    customer = st.number_input(label="Customer(Min:17.21910, Max:17.23015)", format="%0.15f")
    status = st.number_input(label="**Enter the Value for STATUS**/ Min:0.0, Max:8.0")
    thickness = st.number_input(label="Thickness (Min:-1.71479, Max:3.28154)", format="%0.15f")
    item_type = st.number_input(label="Item type (Min:1.0, Max:7.0)")
    application = st.number_input(label="Application(Min:2.0, Max:87.5)")
    width = st.number_input(label="Width(Min:700.0, Max:1980.0)")
    product_ref = st.number_input(label="Product_ref(Min:611728, Max:1722207579)")
    date_differ = st.number_input(label="Datedifference(Min:0, Max:689)")
    st.write("Min,Max values were given for Refference")

    button = st.button(":green[Predict the SellingPrice]", use_container_width=True)

    if button:
        price = predict_selling_price(customer, status, item_type, application, thickness, width, product_ref, date_differ)
        st.write("## :red[**The Selling Price is :**]", price)
