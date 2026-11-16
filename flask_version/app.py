
import joblib
import pandas as pd

from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved model package
bundle = joblib.load("crop_yield_model.joblib")

model = bundle["model"]

feature_columns = bundle["feature_columns"]

states = bundle["states"]

crops = bundle["crops"]

seasons = bundle["seasons"]

best_model_name = bundle["best_model_name"]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    error = None

    form_data = {}

    if request.method == "POST":

        form_data = request.form.to_dict()

        try:

            year = int(request.form["year"])

            state = request.form["state"]

            crop = request.form["crop"]

            season = request.form["season"]

            area = float(request.form["area"])

            annual_rainfall = float(
                request.form["annual_rainfall"]
            )

            fertilizer = float(
                request.form["fertilizer"]
            )

            pesticide = float(
                request.form["pesticide"]
            )

            avg_temperature = float(
                request.form["avg_temperature"]
            )

            avg_humidity = float(
                request.form["avg_humidity"]
            )

            weather_rainfall = float(
                request.form["weather_rainfall"]
            )

            if year < 2000 or year > 2020:
                raise ValueError(
                    "Year must be between 2000 and 2020."
                )

            if state not in states:
                raise ValueError("Invalid state selected.")

            if crop not in crops:
                raise ValueError("Invalid crop selected.")

            if season not in seasons:
                raise ValueError("Invalid season selected.")

            if area <= 0:
                raise ValueError("Area must be greater than zero.")

            if min(
                annual_rainfall,
                fertilizer,
                pesticide,
                avg_humidity,
                weather_rainfall
            ) < 0:
                raise ValueError(
                    "Agricultural values cannot be negative."
                )

            if not 0 <= avg_humidity <= 100:
                raise ValueError(
                    "Humidity must be between 0 and 100."
                )

            if not -5 <= avg_temperature <= 50:
                raise ValueError(
                    "Temperature must be between -5 and 50."
                )

            # Feature engineering must match training
            fertilizer_per_ha = fertilizer / area

            pesticide_per_ha = pesticide / area

            input_data = {

                "Year": year,

                "Area": area,

                "Annual_Rainfall": annual_rainfall,

                "Fertilizer": fertilizer,

                "Pesticide": pesticide,

                "Avg_Temperature": avg_temperature,

                "Avg_Humidity": avg_humidity,

                "Weather_Rainfall": weather_rainfall,

                "Fertilizer_per_ha": fertilizer_per_ha,

                "Pesticide_per_ha": pesticide_per_ha,

                "State": state,

                "Crop": crop,

                "Season": season

            }

            input_df = pd.DataFrame(
                [input_data],
                columns=feature_columns
            )

            result = model.predict(input_df)[0]

            prediction = round(float(result), 4)

        except (ValueError, KeyError) as e:

            error = str(e)

        except Exception:

            error = (
                "Prediction could not be completed. "
                "Please check your input values."
            )

    return render_template(

        "index.html",

        states=states,

        crops=crops,

        seasons=seasons,

        prediction=prediction,

        error=error,

        form_data=form_data,

        model_name=best_model_name

    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
