from flask import Blueprint, request, jsonify
from app.models.utils import perform_forecasting

bp = Blueprint('main', __name__)

@bp.route('/forecast', methods=['POST'])
def forecast_data():
    """API endpoint to perform forecasting on incoming retail data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        forecast_periods = request.args.get('periods', default=30, type=int)
        results = perform_forecasting(data, forecast_periods)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500 