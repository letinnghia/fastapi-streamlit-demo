import requests
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def get_numerical_cols(df):
    # Nhận vào 1 dataframe, trả về danh sách tên những cột chứa giá trị đếm được (interger, float)
    pass

def load_data():
    # Dùng streamlit tạo box cho người dùng upload file .csv,
    # Nếu có file tải lên thì trả về 1 dataframe từ file .csv đó.
    # Trả về None nếu không có file .csv nào được upload.
    pass


def show_data(df):
    # Nhận vào 1 dataframe, dùng streamlit hiển thị dataframe đó
    pass


def run_linear_regression(df):
    # Nhận vào 1 dataframe, dùng streamlit kết hợp fastAPI, viết hàm chạy linear regression như sau:
    # - Cho người dùng nhập input là 1 mảng 2 chiều X (independent variables)
    # và 1 mảng 1 chiều y (dependent variable) từ dataframe. cả 2 đều phải chứa giá
    # trị đếm được (sủ dụng hàm get_numerical_cols đã viết bên trên).
    # - Cho người dùng chọn độ lớn của tập test.
    # - Từ input: X, y, test_sz, gọi api sử dụng fastAPI đã viết trước đó
    # và hiển thị những thông số quan trọng: coefficent, itercept, score 
    # - Truyền những tham số cần thiết vào hàm visualize_linear_regression
    pass


def visualize_linear_regression(X, y, y_predicted, X_label, y_label):
    X = np.array(X)
    y = np.array(y)
    y_predicted = np.array(y_predicted)

    fig = plt.figure(figsize=(20, 12))

    if len(X[0]) == 1:
        plt.scatter(X, y, s=10)
        plt.plot(X.reshape(-1, 1), y_predicted.reshape(-1, 1), color="r")
        plt.xlabel(X_label[0])
        plt.ylabel(y_label)

    if len(X[0]) == 2:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1,1,1, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], y)
        ax.set_xlabel(X_label[0], fontsize=5)
        ax.set_ylabel(X_label[1], fontsize=5)
        ax.set_zlabel(y_label, fontsize=5)
        ax.tick_params(axis="both", labelsize=6, length=5)
        ax.plot_surface(X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1), y_predicted.reshape(-1, 1), color="r")
    
    st.write(fig, width=100, height=100)


if __name__ == "__main__":
    st.set_page_config(page_title="Linear Regression", page_icon=":smile:", layout="wide")
    st.title(":blue[LINEAR REGRESSION PLAYGROUND]")
    df = load_data()
    if df is not None:
        show_data(df)
        run_linear_regression(df)

        
    