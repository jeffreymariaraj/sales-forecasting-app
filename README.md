# 📈 Sales Forecasting App

A modern, interactive web application for forecasting sales using machine learning and visualizing results with beautiful charts.

![Forecasting Demo](https://img.shields.io/badge/Powered%20By-Python%20%7C%20Flask%20%7C%20MySQL%20%7C%20XGBoost-blue)

---

## 🚀 Features

- **Connects to MySQL**: Securely fetches sales data from your MySQL database.
- **Machine Learning Forecasts**: Uses XGBoost and other models for accurate time series forecasting.
- **Interactive Web Interface**: Upload CSV, query SQL, or input data directly.
- **Beautiful Visualizations**: Highcharts-powered dynamic charts for actual and forecasted sales.
- **JSON Debug View**: See the raw API response for easy debugging.
- **Configurable & Secure**: Uses environment variables for all sensitive settings.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, pandas, XGBoost
- **Database**: MySQL (`mysql-connector-python`, `pyodbc`)
- **Frontend**: HTML, CSS, Bootstrap, jQuery, Highcharts
- **Configuration**: `.env` file for secrets and DB credentials

---

## ⚡ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/jeffreymariaraj/sales-forecasting-app.git
   cd sales-forecasting-app
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables**
   - Copy `.env.example` to `.env` and fill in your DB credentials.
4. **Run the app**
   ```bash
   python run.py
   # or
   flask run
   ```
5. **Open in your browser**
   - Go to [http://localhost:5000](http://localhost:5000)

---

## 🖥️ Usage

- **Direct Input**: Paste or generate sample data in JSON format.
- **SQL Input**: Query your MySQL database directly from the UI.
- **CSV Upload**: Upload a CSV file to forecast sales.
- **View Results**: See interactive charts and inspect the raw JSON API output.

---

## 🔒 Security
- All credentials are managed via environment variables.
- Never commit your actual `.env` file with secrets to the repo.

---

## 📚 Project Structure

```
forecast-api-2/
├── app/
│   ├── models/           # Machine learning models (XGBoost, LSTM, etc.)
│   ├── routes.py         # Flask API routes
│   └── __init__.py
├── templates/
│   └── index.html        # Main web interface
├── database.py           # Database connection and data fetching
├── main.py               # App entry point
├── run.py                # Alternative app runner
├── .env                  # Environment variables (not committed)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙋‍♂️ Author

**Jeffrey Maria Raj**  
[GitHub](https://github.com/jeffreymariaraj)

---

> Made with ❤️ for robust, insightful sales forecasting.
