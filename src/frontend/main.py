import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

st.title("LaLiga Match Predictions")
st.write("Predict match outcomes, goals, corners, and yellow cards for La Liga matches")

# Sidebar for input parameters
st.sidebar.header('Input Parameters')

def user_input_features():
    jornada = st.sidebar.number_input("Jornada", min_value=1, max_value=38, value=1)
    return {"jornada": jornada}

def display_predictions(predictions_df):
    # Display basic match predictions
    st.subheader("Match Predictions")
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    for _, row in predictions_df.iterrows():
        with col1:
            st.write(f"### {row['Anfitrion']} vs {row['Adversario']}")
            st.write(f"Win probability: {row['Probabilidad_Victoria']:.2%}")
            st.write(f"Draw probability: {row['Probabilidad_Empate']:.2%}")
            st.write(f"Loss probability: {row['Probabilidad_Derrota']:.2%}")
            st.write("---")
        
        with col2:
            st.write("### Goals")
            st.write(f"Home: {row['Goles_Predichos_Local']:.1f} ({row['Goles_Predichos_Local_CI_Lower']:.1f}-{row['Goles_Predichos_Local_CI_Upper']:.1f})")
            st.write(f"Away: {row['Goles_Predichos_Visitante']:.1f} ({row['Goles_Predichos_Visitante_CI_Lower']:.1f}-{row['Goles_Predichos_Visitante_CI_Upper']:.1f})")
            st.write("---")
        
        with col3:
            st.write("### Other Stats")
            st.write(f"Corners Home: {row['Corners_Predichos_Local']:.1f} ({row['Corners_Predichos_Local_CI_Lower']:.1f}-{row['Corners_Predichos_Local_CI_Upper']:.1f})")
            st.write(f"Corners Away: {row['Corners_Predichos_Visitante']:.1f} ({row['Corners_Predichos_Visitante_CI_Lower']:.1f}-{row['Corners_Predichos_Visitante_CI_Upper']:.1f})")
            st.write(f"Yellow Cards Home: {row['Amarillas_Predichas_Local']:.1f} ({row['Amarillas_Predichas_Local_CI_Lower']:.1f}-{row['Amarillas_Predichas_Local_CI_Upper']:.1f})")
            st.write(f"Yellow Cards Away: {row['Amarillas_Predichas_Visitante']:.1f} ({row['Amarillas_Predichas_Visitante_CI_Lower']:.1f}-{row['Amarillas_Predichas_Visitante_CI_Upper']:.1f})")
            st.write("---")

def create_visualization(predictions_df, metric_type='goals'):
    st.subheader(f"{metric_type.title()} Visualization")
    
    fig = go.Figure()
    
    if metric_type == 'goals':
        home_col = 'Goles_Predichos_Local'
        away_col = 'Goles_Predichos_Visitante'
        home_ci_lower = 'Goles_Predichos_Local_CI_Lower'
        home_ci_upper = 'Goles_Predichos_Local_CI_Upper'
        away_ci_lower = 'Goles_Predichos_Visitante_CI_Lower'
        away_ci_upper = 'Goles_Predichos_Visitante_CI_Upper'
    elif metric_type == 'corners':
        home_col = 'Corners_Predichos_Local'
        away_col = 'Corners_Predichos_Visitante'
        home_ci_lower = 'Corners_Predichos_Local_CI_Lower'
        home_ci_upper = 'Corners_Predichos_Local_CI_Upper'
        away_ci_lower = 'Corners_Predichos_Visitante_CI_Lower'
        away_ci_upper = 'Corners_Predichos_Visitante_CI_Upper'
    else:  # yellow cards
        home_col = 'Amarillas_Predichas_Local'
        away_col = 'Amarillas_Predichas_Visitante'
        home_ci_lower = 'Amarillas_Predichas_Local_CI_Lower'
        home_ci_upper = 'Amarillas_Predichas_Local_CI_Upper'
        away_ci_lower = 'Amarillas_Predichas_Visitante_CI_Lower'
        away_ci_upper = 'Amarillas_Predichas_Visitante_CI_Upper'
    
    # Add home team predictions
    fig.add_trace(go.Bar(
        name='Home Team',
        x=[f"{row['Anfitrion']} vs {row['Adversario']}" for _, row in predictions_df.iterrows()],
        y=predictions_df[home_col],
        error_y=dict(
            type='data',
            symmetric=False,
            array=predictions_df[home_ci_upper] - predictions_df[home_col],
            arrayminus=predictions_df[home_col] - predictions_df[home_ci_lower]
        )
    ))
    
    # Add away team predictions
    fig.add_trace(go.Bar(
        name='Away Team',
        x=[f"{row['Anfitrion']} vs {row['Adversario']}" for _, row in predictions_df.iterrows()],
        y=predictions_df[away_col],
        error_y=dict(
            type='data',
            symmetric=False,
            array=predictions_df[away_ci_upper] - predictions_df[away_col],
            arrayminus=predictions_df[away_col] - predictions_df[away_ci_lower]
        )
    ))
    
    fig.update_layout(
        barmode='group',
        title=f'Predicted {metric_type.title()} with Confidence Intervals',
        xaxis_title='Match',
        yaxis_title=f'Predicted {metric_type.title()}',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    input_dict = user_input_features()
    
    if st.sidebar.button('Predict'):
        try:
            # Make prediction request
            response = requests.post(
                "http://backend:8000/predict",
                json=input_dict,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions_df = pd.DataFrame(data['predictions'])
                
                # Display predictions
                display_predictions(predictions_df)
                
                # Add tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Goals", "Corners", "Yellow Cards"])
                
                with tab1:
                    create_visualization(predictions_df, 'goals')
                
                with tab2:
                    create_visualization(predictions_df, 'corners')
                
                with tab3:
                    create_visualization(predictions_df, 'yellow cards')
                
                # Display model version
                st.sidebar.info(f"Model Version: {data['model_version']}")
                
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the prediction service: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()