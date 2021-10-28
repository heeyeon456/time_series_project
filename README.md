# time_series_project
---

## Task
- Day-ahead PV power Forecasting

## Input
- DKASC Dataset (Alice Springs)
http://dkasolarcentre.com.au/download?location=alice-springs
- Additional Features(weather features):
    * temperature, humidity, Global Horizonal Radiation, Diffused Horizontal Radiation, rain

## Output
- Day ahead PV power prediction result

## Models
- Scikit-learn based: models/skmodels.py
- Pytorch: models/mlp.py, tcn.py, lstm.py, cnn_lstm.py

## Usage
* ML-based model
python engine_ml.py --phase=train --model_to_use=mlp --lag=13 --additional_feat="rain"

* DL-based model
python engine_dl.py --phase=train --model_to_use=mlp --lag=13 --additional_feat="rain"
