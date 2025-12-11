from fastapi import FastAPI,File, UploadFile
from model_helper import predict

app = FastAPI()

@app.post("/predict")

async def get_prediction(file: UploadFile = File(...)):    #file type will by upload type and = file(...) means its mandatory param
    try:

        image_bytes = await file.read()                         #stored uploaded image in b

        image_path = "temp_file.jpeg"                           #stored a temp image will overwrite with uploaded img
        with open(image_path,"wb") as f:
            f.write(image_bytes)

        prediction = predict(image_path)

        return {"prediction": prediction}                   #returning response in json format
    except Exception as e:
        return {"error": str(e)}