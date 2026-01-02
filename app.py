from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run
from typing import Optional
from pathlib import Path
import os

# Importing constants and pipeline modules from the project (update these if your classes have different names)
from src.constants import APP_HOST, APP_PORT
from src.pipeline.prediction_pipeline import VehicleData, VehicleDataClassifier  # <-- Renamed to match your context (e.g., bike rental demand)
from src.pipeline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Determine project base dir and mount static/templates with absolute paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the bike rental-related attributes expected from the form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.Hour: Optional[int] = None
        self.Temperature: Optional[float] = None
        self.Humidity: Optional[int] = None
        self.Wind_speed: Optional[float] = None
        self.Visibility: Optional[int] = None
        self.dew_point_temperature: Optional[float] = None
        self.Solar_Radiation: Optional[float] = None
        self.Rainfall: Optional[float] = None
        self.snowfall: Optional[float] = None
        self.month: Optional[int] = None
        self.day: Optional[int] = None
        self.Seasons_Autumn: Optional[int] = None
        self.Seasons_Spring: Optional[int] = None
        self.Seasons_Summer: Optional[int] = None
        self.Seasons_Winter: Optional[int] = None
        self.Holiday_No_Holiday: Optional[int] = None
        self.Functioning_Day_Yes: Optional[int] = None
               
    async def get_bike_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()

        def to_int(value, default=0):
            try:
                return int(value)
            except Exception:
                return default

        def to_float(value, default=0.0):
            try:
                return float(value)
            except Exception:
                return default

        self.Hour = to_int(form.get("Hour"))
        self.Temperature = to_float(form.get("Temperature"))
        self.Humidity = to_int(form.get("Humidity"))
        self.Wind_speed = to_float(form.get("Wind_speed"))
        self.Visibility = to_int(form.get("Visibility"))
        self.dew_point_temperature = to_float(form.get("dew_point_temperature"))
        self.Solar_Radiation = to_float(form.get("Solar_Radiation"))
        self.Rainfall = to_float(form.get("Rainfall"))
        self.snowfall = to_float(form.get("snowfall"))
        self.month = to_int(form.get("month"))
        self.day = to_int(form.get("day"))

        # categorical/binary flags (0 or 1)
        self.Seasons_Autumn = to_int(form.get("Seasons_Autumn"))
        self.Seasons_Spring = to_int(form.get("Seasons_Spring"))
        self.Seasons_Summer = to_int(form.get("Seasons_Summer"))
        self.Seasons_Winter = to_int(form.get("Seasons_Winter"))
        self.Holiday_No_Holiday = to_int(form.get("Holiday_No_Holiday"))
        self.Functioning_Day_Yes = to_int(form.get("Functioning_Day_Yes"))

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for bike rental data input.
    """
    return templates.TemplateResponse(
            "data.html", {"request": request, "context": "Rendering"})  # use existing template file

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_bike_data()
       
        bike_data = VehicleData(  # <-- Update class name to match your prediction_pipeline
            Hour=form.Hour,
            Temperature=form.Temperature,
            Humidity=form.Humidity,
            Wind_speed=form.Wind_speed,
            Visibility=form.Visibility,
            dew_point_temperature=form.dew_point_temperature,
            Solar_Radiation=form.Solar_Radiation,
            Rainfall=form.Rainfall,
            snowfall=form.snowfall,
            month=form.month,
            day=form.day,
            Seasons_Autumn=form.Seasons_Autumn,
            Seasons_Spring=form.Seasons_Spring,
            Seasons_Summer=form.Seasons_Summer,
            Seasons_Winter=form.Seasons_Winter,
            Holiday_No_Holiday=form.Holiday_No_Holiday,
            Functioning_Day_Yes=form.Functioning_Day_Yes
        )
        
        # Convert form data into a DataFrame for the model
        bike_df = bike_data.get_vehicle_input_data_frame()
        
        # Initialize the prediction pipeline
        model_predictor = VehicleDataClassifier()  # <-- Update class name
        
        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=bike_df)[0]
        
        # Interpret the prediction result (adjust based on your model's output, e.g., predicted rented bike count)
        status = f"Predicted Rented Bike Count: {value}"  # <-- Customize this display
        
        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "data.html",
            {"request": request, "context": status},
        )
       
    except Exception as e:
        return {"status": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)