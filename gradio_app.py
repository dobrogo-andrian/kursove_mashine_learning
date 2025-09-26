import gradio as gr
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pickle
import os

models = {}
feature_columns = []
#csv_path = os.path.join('.', 'cleaned_df', 'cleaned_df.csv')
csv_path = './cleaned_df/cleaned_df.csv'
def load_df(df_path):
    print(f"✅ Завантаження датасету")
    return pd.read_csv(df_path)


def prepare_features(df, lags=[2, 4]):
    """Підготовка ознак з лагами"""
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['average_salary'].shift(lag)
    df.dropna(inplace=True)
    return df


def train_ml_model_for_seniority(df_cleaned, seniority, lags=[2]):
    df_sub = df_cleaned[df_cleaned['final_seniority'] == seniority].copy()
    df_sub = prepare_features(df_sub, lags=lags)

    target = 'average_salary'
    lag_features = [f'lag_{lag}' for lag in lags]
    features = ['gdp_mln_usd', 'exchange_rate_uah_usd', 'google_search_for_data_science', 
                     'final_consumption_expenditure', 'delta_gdp_mln_usd', 'percent_delta_gdp_mln_usd',
                     'consumer price index', 'average_salary_usd', 'delta_average_salary_usd',
                     'percent_delta_average_salary_usd', 'year',	'half', 'cleaned'] + lag_features

    train_df = df_sub.iloc[:-4]
    test_df = df_sub.iloc[-4:]

    X_train = train_df[features]
    y_train = train_df[target]

    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)

    return model

def train_initial_models(df):
    """Тренування початкових моделей для демонстрації"""
    
    print(f"✅ Початок тренування")
    print(f"{df.head()}")

    for level in df['final_seniority'].unique():
        models[level] = train_ml_model_for_seniority(df, level)
        print(f"✅ Модель для {level} натренована")
    

def predict_salary(seniority, gdp_mln_usd, exchange_rate, google_search, consumption_expenditure,
                  delta_gdp, percent_delta_gdp, cpi, avg_salary_usd, delta_salary_usd,
                  percent_delta_salary_usd, year, half, cleaned, lag_2):
    """Передбачення зарплати на основі вхідних параметрів"""
    
    global models, feature_columns
    
    if not models:
        return "❌ Моделі не натреновані. Спочатку ініціалізуйте додаток."
    
    if seniority not in models:
        available = ', '.join(models.keys())
        return f"❌ Модель для '{seniority}' не знайдена. Доступні: {available}"
    
    try:
        # Створення DataFrame з вхідними даними у правильному порядку
        # Порядок має відповідати тому, що очікує модель
        input_data = {
            'gdp_mln_usd': [gdp_mln_usd],
            'delta_gdp_mln_usd': [delta_gdp],
            'lag_2': [lag_2],
            'year': [year],
            'half': [half],
            'final_consumption_expenditure': [consumption_expenditure],
            'cleaned': [cleaned],
            'delta_average_salary_usd': [delta_salary_usd],
            'percent_delta_average_salary_usd': [percent_delta_salary_usd],
            'consumer price index': [cpi],
            'percent_delta_gdp_mln_usd': [percent_delta_gdp],
            'average_salary_usd': [avg_salary_usd],
            'exchange_rate_uah_usd': [exchange_rate],
            'google_search_for_data_science': [google_search]
        }
        
        df_input = pd.DataFrame(input_data)
        
        model = models[seniority]
        
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        else:
            expected_features = [
                'gdp_mln_usd', 'delta_gdp_mln_usd', 'lag_2', 'year', 'half', 
                'final_consumption_expenditure', 'cleaned', 'delta_average_salary_usd', 
                'percent_delta_average_salary_usd', 'consumer price index', 
                'percent_delta_gdp_mln_usd', 'average_salary_usd', 'exchange_rate_uah_usd', 
                'google_search_for_data_science'
            ]
        
        missing_features = [col for col in expected_features if col not in df_input.columns]
        if missing_features:
            return f"❌ Відсутні ознаки: {', '.join(missing_features)}"
        
        X_input = df_input[expected_features]
        
        prediction = model.predict(X_input)[0]

        result = f"""
🎯 **Прогноз зарплати для {seniority.upper()}:**

💰 **Прогнозована зарплата:** {prediction:,.0f} USD
📊 **Курс долара:** {exchange_rate:.2f} UAH/USD

📈 **Вхідні параметри:**
• GDP: {gdp_mln_usd:,} млн USD
• Пошуки "data science": {google_search}
• Споживчі витрати: {consumption_expenditure:,} млн USD
• Індекс споживчих цін: {cpi:.1f}
• Рік/півріччя: {year}/{half}
• Лаг 2 періоди: {lag_2:,.0f} UAH
"""
        
        return result
        
    except Exception as e:
        return f"❌ Помилка при передбаченні: {str(e)}"

def create_gradio_app():
    """Створення Gradio додатку для передбачення зарплат"""

    with gr.Blocks(title="💰 Salary Predictor", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 💰 Прогнозування зарплат в IT
        
        Введіть економічні параметри та отримайте прогноз зарплати Data science спеціалістів для різних рівнів seniority
        """)
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Параметри для прогнозу")
                
                seniority = gr.Dropdown(
                    choices=["junior", "middle", "senior"],
                    label="Рівень Seniority",
                    value="middle"
                )
                
                with gr.Row():
                    year = gr.Number(label="Рік", value=2024, precision=0)
                    half = gr.Dropdown(choices=[1, 2], label="Півріччя", value=1)
                
                gdp = gr.Number(label="GDP (млн USD)", value=153781)
                exchange_rate = gr.Number(label="Курс UAH/USD", value=41.5)
                google_search = gr.Number(label="Google пошуки 'data science'", value=75, precision=0)
                consumption = gr.Number(label="Споживчі витрати (млн USD)", value=150000)
                lag_2 = gr.Number(label="Лаг 2 (USD)", value=1160)
                with gr.Row():
                    delta_gdp = gr.Number(label="Δ GDP", value=0)
                    percent_delta_gdp = gr.Number(label="% Δ GDP", value=2.5)
                
                cpi = gr.Number(label="Індекс споживчих цін", value=110)
                
                with gr.Row():
                    avg_salary_usd = gr.Number(label="Сер. зарплата (USD)", value=1200)
                    delta_salary_usd = gr.Number(label="Δ Зарплата (USD)", value=50)
                
                percent_delta_salary = gr.Number(label="% Δ Зарплата (USD)", value=4)
                cleaned = gr.Dropdown(choices=[0, 1], label="Очищені дані", value=1)
                
                predict_btn = gr.Button("🎯 Передбачити зарплату", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Результат прогнозу")
            prediction_output = gr.Markdown(value="Введіть параметри та натисніть 'Передбачити зарплату'")

        predict_btn.click(
            fn=predict_salary,
            inputs=[seniority, gdp, exchange_rate, google_search, consumption,
                   delta_gdp, percent_delta_gdp, cpi, avg_salary_usd, delta_salary_usd,
                   percent_delta_salary, year, half, cleaned, lag_2],
            outputs=prediction_output
        )
        
        gr.Markdown("""
        ---
        ### 📋 Інструкції:
        1. **Ініціалізуйте** модель демо-даними або завантажте власний CSV
        2. **Введіть параметри** економічного стану та попередні зарплати
        3. **Виберіть seniority** рівень (junior/middle/senior)
        4. **Натисніть передбачити** для отримання прогнозу
        
        Модель враховує макроекономічні показники та історичні дані зарплат.
        """)
    
    return app

if __name__ == "__main__":
    df = load_df(csv_path)
    print(df.head())
    train_initial_models(df) 
    app = create_gradio_app()
    app.launch(
        share=True,
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True
    )