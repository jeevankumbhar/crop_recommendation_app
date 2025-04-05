import streamlit as st
import numpy as np
import pandas as pd
from model.crop_recommendation_model import CropRecommender
from utils.visualization import create_gauge_chart, create_feature_importance_plot, create_model_comparison_plot
from utils.pdf_generator import create_prediction_pdf
from utils.advanced_features import IrrigationScheduler, EconomicAnalyzer, CropRotationPlanner
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Load custom CSS
with open('styles/custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize the model and advanced features
@st.cache_resource
def load_model():
    return CropRecommender(), IrrigationScheduler(), EconomicAnalyzer(), CropRotationPlanner()

try:
    model, irrigation_scheduler, economic_analyzer, rotation_planner = load_model()
    # Load dataset for range calculation
    df = pd.read_csv("attached_assets/Crop_recommendation (1).csv")
except Exception as e:
    st.error("Failed to initialize the system. Please try again later.")
    st.stop()

# Header section
st.title("🌾 Crop Recommendation System")
st.markdown("""
This system helps farmers make informed decisions about crop selection based on soil conditions 
and environmental factors. Enter your parameters below to get personalized recommendations.
""")

# Create tabs for different features
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Crop Recommendation", 
    "Irrigation Schedule", 
    "Economic Analysis",
    "Crop Rotation",
    "Database View"
])

with tab1:
    # Original crop recommendation interface
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Soil Parameters")
        nitrogen = st.slider("Nitrogen (N) mg/kg", 
                            float(df['N'].min()), float(df['N'].max()), 
                            float(df['N'].mean()))
        phosphorus = st.slider("Phosphorus (P) mg/kg", 
                              float(df['P'].min()), float(df['P'].max()), 
                              float(df['P'].mean()))
        potassium = st.slider("Potassium (K) mg/kg", 
                             float(df['K'].min()), float(df['K'].max()), 
                             float(df['K'].mean()))
        ph = st.slider("pH value", 
                       float(df['ph'].min()), float(df['ph'].max()), 
                       float(df['ph'].mean()))

    with col2:
        st.subheader("Environmental Conditions")
        temperature = st.slider("Temperature (°C)", 
                              float(df['temperature'].min()), float(df['temperature'].max()), 
                              float(df['temperature'].mean()))
        humidity = st.slider("Humidity (%)", 
                            float(df['humidity'].min()), float(df['humidity'].max()), 
                            float(df['humidity'].mean()))
        rainfall = st.slider("Rainfall (mm)", 
                            float(df['rainfall'].min()), float(df['rainfall'].max()), 
                            float(df['rainfall'].mean()))

    if st.button("Get Crop Recommendation"):
        with st.spinner("Analyzing parameters..."):
            try:
                # Prepare input features
                features = np.array([nitrogen, phosphorus, potassium, 
                                   temperature, humidity, ph, rainfall])

                # Get prediction and probabilities
                prediction, probabilities = model.predict(features)

                # Display results
                st.success(f"### Recommended Crop: {prediction.title()}")

                # Create three columns for gauge charts
                g1, g2, g3 = st.columns(3)

                with g1:
                    st.plotly_chart(create_gauge_chart(
                        nitrogen, "Nitrogen Level", 
                        df['N'].min(), df['N'].max()
                    ))
                with g2:
                    st.plotly_chart(create_gauge_chart(
                        ph, "pH Level", 
                        df['ph'].min(), df['ph'].max()
                    ))
                with g3:
                    st.plotly_chart(create_gauge_chart(
                        rainfall, "Rainfall", 
                        df['rainfall'].min(), df['rainfall'].max()
                    ))

                # Show model performance metrics
                st.subheader("Model Performance Analysis")
                model_scores = model.get_model_scores()

                # Create columns for metrics
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("#### Model Accuracy Comparison")
                    model_comparison_fig = create_model_comparison_plot(model_scores)
                    st.plotly_chart(model_comparison_fig)

                with c2:
                    st.markdown("#### Detailed Model Metrics")
                    for model_name, scores in model_scores.items():
                        st.markdown(f"""
                        **{model_name.title()}**:
                        - Accuracy: {scores['accuracy']:.2%}
                        - Cross-validation Score: {scores['cv_mean']:.2%} (±{scores['cv_std']:.2%})
                        """)

                # Show top 3 predictions with probabilities
                st.subheader("Top Crop Recommendations")
                crop_labels = model.get_crop_labels()
                top_3_idx = probabilities.argsort()[-3:][::-1]

                crop_probs = []
                for idx in top_3_idx:
                    crop = crop_labels[idx]
                    prob = probabilities[idx]
                    crop_probs.append((crop, prob * 100))
                    st.markdown(f"""
                    - **{crop.title()}**: {prob*100:.1f}% confidence
                    """)

                # Feature importance plot
                st.subheader("Parameter Importance Analysis")
                importance_scores = model.get_feature_importance()
                feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 
                                   'Temperature', 'Humidity', 'pH', 'Rainfall']
                feature_importance_fig = create_feature_importance_plot(
                    importance_scores, feature_names
                )
                st.plotly_chart(feature_importance_fig)

                # Prepare data for PDF
                prediction_data = {
                    'prediction': prediction,
                    'parameters': {
                        'Nitrogen': nitrogen,
                        'Phosphorus': phosphorus,
                        'Potassium': potassium,
                        'Temperature': temperature,
                        'Humidity': humidity,
                        'pH': ph,
                        'Rainfall': rainfall
                    },
                    'model_scores': model_scores
                }

                # Generate PDF
                pdf_output = create_prediction_pdf(
                    prediction_data,
                    feature_importance_fig,
                    crop_probs
                )

                # Add download button
                st.download_button(
                    label="📄 Download Recommendation Report (PDF)",
                    data=pdf_output,
                    file_name="crop_recommendation_report.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"An error occurred while making the prediction: {str(e)}")

with tab2:
    st.subheader("Irrigation Scheduling")

    # Get available crops from the irrigation scheduler database
    available_crops_irrigation = list(irrigation_scheduler.water_requirements.keys())
    available_crops_irrigation.sort()

    # Input fields for irrigation
    crop_name = st.selectbox("Select crop", available_crops_irrigation)
    area = st.number_input("Field area (hectares)", min_value=0.1, value=1.0)
    monthly_rainfall = st.number_input("Expected monthly rainfall (mm)", min_value=0.0, value=100.0)

    if st.button("Calculate Irrigation Schedule"):
        if crop_name:
            schedule = irrigation_scheduler.calculate_schedule(crop_name, area, monthly_rainfall)
            if schedule:
                st.success("Irrigation Schedule Generated")
                schedule_df = pd.DataFrame(schedule)
                st.dataframe(schedule_df)

                # Create downloadable CSV
                csv = schedule_df.to_csv(index=False)
                st.download_button(
                    label="Download Schedule (CSV)",
                    data=csv,
                    file_name="irrigation_schedule.csv",
                    mime="text/csv"
                )
            else:
                st.error("Crop not found in database")

with tab3:
    st.subheader("Economic Analysis")

    # Get available crops from the economic analyzer database
    available_crops_economics = list(economic_analyzer.crop_economics.keys())
    available_crops_economics.sort()

    # Input fields for economic analysis
    analysis_crop = st.selectbox("Select crop for analysis", available_crops_economics)
    analysis_area = st.number_input("Field area for analysis (hectares)", min_value=0.1, value=1.0)

    if st.button("Analyze Economics"):
        if analysis_crop:
            analysis = economic_analyzer.analyze_crop(analysis_crop, analysis_area)
            if analysis:
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Cost (₹)", f"₹{analysis['total_cost']:,.2f}")
                    st.metric("Expected Yield (kg)", f"{analysis['expected_yield']:,.2f}")

                with col2:
                    st.metric("Expected Revenue (₹)", f"₹{analysis['expected_revenue']:,.2f}")
                    st.metric("ROI", f"{analysis['roi_percentage']}%")

                st.metric("Expected Profit (₹)", f"₹{analysis['expected_profit']:,.2f}")
            else:
                st.error("Crop not found in database")

with tab4:
    st.subheader("Crop Rotation Planning")

    # Create a comprehensive list of all crops from all data sources
    all_seasonal_crops = []
    # Get crops from rotation planner
    for season_crops in rotation_planner.seasonal_crops.values():
        all_seasonal_crops.extend(season_crops)
    # Get crops from irrigation scheduler
    all_seasonal_crops.extend(irrigation_scheduler.water_requirements.keys())
    # Get crops from economic analyzer
    all_seasonal_crops.extend(economic_analyzer.crop_economics.keys())
    all_seasonal_crops = sorted(list(set(all_seasonal_crops)))  # Remove duplicates and sort

    # Input fields for rotation planning
    current_crop = st.selectbox("Select current crop", all_seasonal_crops)
    season = st.selectbox("Select season", ["Summer", "Winter", "Monsoon"])

    if st.button("Get Rotation Suggestions"):
        if current_crop:
            rotation = rotation_planner.suggest_rotation(current_crop, season)
            if rotation:
                st.success("Rotation Suggestions Generated")
                st.write("Suggested crops for next season:")
                for crop in rotation['suggested_crops']:
                    st.markdown(f"- {crop.title()}")
                st.info(rotation['rotation_benefits'])
            else:
                st.error("Could not generate rotation suggestions")

# Footer
st.markdown("""
---
Made with ❤️ for farmers | Data-driven agriculture for better yields
""")

with tab5:
    st.subheader("Crop Database Information")
    
    db_view_option = st.radio(
        "Select Database to View",
        ["Irrigation Requirements", "Economic Analysis", "Crop Rotation"]
    )
    
    if db_view_option == "Irrigation Requirements":
        st.write("### Irrigation Requirements Database")
        
        # Convert irrigation scheduler data to DataFrame
        irrigation_data = []
        for crop, details in irrigation_scheduler.water_requirements.items():
            irrigation_data.append({
                "Crop": crop.title(),
                "Water Need (mm/season)": details['water_need'],
                "Irrigation Frequency (days)": details['frequency']
            })
        
        irrigation_df = pd.DataFrame(irrigation_data)
        irrigation_df = irrigation_df.sort_values("Crop")
        
        st.dataframe(irrigation_df)
        
        # Add download button for irrigation data
        csv = irrigation_df.to_csv(index=False)
        st.download_button(
            label="Download Irrigation Database (CSV)",
            data=csv,
            file_name="irrigation_database.csv",
            mime="text/csv"
        )
    
    elif db_view_option == "Economic Analysis":
        st.write("### Economic Analysis Database")
        
        # Convert economic analyzer data to DataFrame
        economic_data = []
        for crop, details in economic_analyzer.crop_economics.items():
            economic_data.append({
                "Crop": crop.title(),
                "Cost per Hectare (₹)": details['cost_per_hectare'],
                "Average Yield (kg/hectare)": details['avg_yield'],
                "Price per kg (₹)": details['price_per_kg'],
                "Estimated Revenue per Hectare (₹)": details['avg_yield'] * details['price_per_kg'],
                "Estimated Profit per Hectare (₹)": (details['avg_yield'] * details['price_per_kg']) - details['cost_per_hectare'],
                "ROI (%)": round(((details['avg_yield'] * details['price_per_kg']) - details['cost_per_hectare']) / details['cost_per_hectare'] * 100, 2)
            })
        
        economic_df = pd.DataFrame(economic_data)
        economic_df = economic_df.sort_values("Crop")
        
        st.dataframe(economic_df)
        
        # Add download button for economic data
        csv = economic_df.to_csv(index=False)
        st.download_button(
            label="Download Economic Database (CSV)",
            data=csv,
            file_name="economic_database.csv",
            mime="text/csv"
        )
    
    else:  # Crop Rotation
        st.write("### Crop Rotation Database")
        
        # Seasonal crops
        st.subheader("Seasonal Crops")
        seasonal_data = []
        for season, crops in rotation_planner.seasonal_crops.items():
            seasonal_data.append({
                "Season": season.title(),
                "Suitable Crops": ", ".join([crop.title() for crop in crops])
            })
        
        seasonal_df = pd.DataFrame(seasonal_data)
        st.dataframe(seasonal_df)
        
        # Rotation benefits
        st.subheader("Crop Categories for Rotation")
        rotation_data = []
        for category, crops in rotation_planner.rotation_benefits.items():
            rotation_data.append({
                "Category": category.title(),
                "Crops": ", ".join([crop.title() for crop in crops]),
                "Rotation Benefit": "Nitrogen fixing" if category == "legumes" else 
                                   "Soil structure improvement" if category == "cereals" else
                                   "Economic value"
            })
        
        rotation_df = pd.DataFrame(rotation_data)
        st.dataframe(rotation_df)
        
        # Add download buttons for rotation data
        csv_seasonal = seasonal_df.to_csv(index=False)
        csv_rotation = rotation_df.to_csv(index=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Seasonal Crops (CSV)",
                data=csv_seasonal,
                file_name="seasonal_crops.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="Download Crop Categories (CSV)",
                data=csv_rotation,
                file_name="crop_categories.csv",
                mime="text/csv"
            )
