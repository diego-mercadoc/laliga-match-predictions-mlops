# LaLiga Predictions Frontend Documentation

## Overview

The frontend application is built with Streamlit and provides an interactive interface for the LaLiga Match Prediction System. It allows users to visualize predictions, historical data, and model performance metrics.

## Components

### 1. Match Prediction Interface
- Team selection dropdowns
- Date picker for match scheduling
- Form and head-to-head data input
- Prediction results visualization
- Confidence score display

### 2. Historical Data Dashboard
- Match history visualization
- Team performance trends
- Head-to-head statistics
- Form analysis charts

### 3. Model Performance Monitoring
- Accuracy metrics visualization
- Feature importance plots
- Model drift monitoring
- Prediction distribution charts

### 4. Data Sources Display
- Historical data (`laliga.csv`) overview
- Current season data (`LaLiga Dataset 2023-2024.xlsx`) summary
- Live data updates from fbref.com
- Data quality metrics

## Configuration

### Environment Variables
```bash
BACKEND_API_URL=http://localhost:8000
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Dependencies
- streamlit
- pandas
- plotly
- requests

## Usage

### Running the Application
```bash
cd src/frontend
streamlit run main.py
```

### Docker Deployment
```bash
docker build -t laliga-frontend .
docker run -p 8501:8501 laliga-frontend
```

## Development Guide

### Adding New Features
1. Create new Streamlit components in `main.py`
2. Follow the existing pattern for API calls
3. Use Streamlit's built-in caching for performance
4. Add error handling and user feedback

### Best Practices
- Use Streamlit's session state for persistence
- Implement proper error handling
- Cache API responses when appropriate
- Follow responsive design principles

### Testing
- Test with different screen sizes
- Verify API integration
- Check error handling
- Validate data visualization

## Troubleshooting

### Common Issues
1. API Connection Problems
   - Check backend service status
   - Verify API URL configuration
   - Check network connectivity

2. Performance Issues
   - Use proper caching
   - Optimize data loading
   - Minimize unnecessary recomputation

3. Display Problems
   - Check browser compatibility
   - Verify screen resolution support
   - Test responsive design

## Security Considerations

- API token management
- Input validation
- Error message sanitization
- Session handling

## Future Improvements

1. Enhanced Visualizations
   - More interactive plots
   - Additional chart types
   - Custom themes

2. User Experience
   - Improved navigation
   - Better mobile support
   - Faster loading times

3. Features
   - Batch predictions
   - Custom reports
   - Data export options 