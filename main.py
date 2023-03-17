from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


app = FastAPI()

class LinearRegressionRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    test_sz: float


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML API!"}

  
@app.post("/linear-regression")
def perform_linear_regression(request: LinearRegressionRequest):
    pass
    # Nhận vào 1 object LinearRegressionRequest được định nghĩa bên trên, hãy:
    # - Phân chia training data, testing data (X_train, y_train và X_test, y_test)
    # - Huấn luyện mô hình linear regression trên tập training data
    # - Chấm điểm mô hình đã được huấn luyện trên tập testing data
    # - Tính một số thông số quan trọng như coefficent, intercept
    # - Tính dự đoán của mô hình trên X_train, gọi là y_predicted
    # Trả về: X_train, y_train, coefficent, intercept, score, y_predicted
    # dưới dạng python dictionary





