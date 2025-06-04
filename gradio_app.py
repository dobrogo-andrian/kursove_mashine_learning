import gradio as gr
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pickle
import os

# Глобальні змінні для зберігання моделей
models = {}
feature_columns = []

def prepare_features(df, lags=[2, 4]):
    """Підготовка ознак з лагами"""
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['average_salary'].shift(lag)
    df.dropna(inplace=True)
    return df

def train_initial_models(df_path=None):
    """Тренування початкових моделей для демонстрації"""
    # Створення демонстраційних даних якщо CSV не надано
    if df_path is None:
        # Генеруємо синтетичні дані для демонстрації
        np.random.seed(42)
        n_samples = 50
        
        data = {
            'final_seniority': np.random.choice(['junior', 'middle', 'senior'], n_samples),
            'gdp_mln_usd': np.random.normal(180000, 20000, n_samples),
            'exchange_rate_uah_usd': np.random.normal(27.5, 2.5, n_samples),
            'google_search_for_data_science': np.random.randint(60, 100, n_samples),
            'final_consumption_expenditure': np.random.normal(150000, 15000, n_samples),
            'delta_gdp_mln_usd': np.random.normal(0, 5000, n_samples),
            'percent_delta_gdp_mln_usd': np.random.normal(2.5, 1.5, n_samples),
            'consumer price index': np.random.normal(110, 10, n_samples),
            'average_salary_usd': np.random.normal(1200, 400, n_samples),
            'delta_average_salary_usd': np.random.normal(50, 100, n_samples),
            'percent_delta_average_salary_usd': np.random.normal(4, 2, n_samples),
            'year': np.random.choice([2022, 2023, 2024], n_samples),
            'half': np.random.choice([1, 2], n_samples),
            'cleaned': np.random.choice([0, 1], n_samples),
            'average_salary': np.random.normal(35000, 10000, n_samples)
        }
        
        # Коригуємо зарплати відповідно до seniority
        for i in range(n_samples):
            if data['final_seniority'][i] == 'junior':
                data['average_salary'][i] = np.random.normal(25000, 5000)
                data['average_salary_usd'][i] = data['average_salary'][i] / data['exchange_rate_uah_usd'][i]
            elif data['final_seniority'][i] == 'middle':
                data['average_salary'][i] = np.random.normal(40000, 8000)
                data['average_salary_usd'][i] = data['average_salary'][i] / data['exchange_rate_uah_usd'][i]
            else:  # senior
                data['average_salary'][i] = np.random.normal(60000, 12000)
                data['average_salary_usd'][i] = data['average_salary'][i] / data['exchange_rate_uah_usd'][i]
        
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(df_path)
    
    global models, feature_columns
    
    # Тренування моделей для кожного seniority
    seniority_levels = df['final_seniority'].unique()
    
    for seniority in seniority_levels:
        print(f"Тренування моделі для {seniority}...")
        
        df_sub = df[df['final_seniority'] == seniority].copy()
        df_sub = prepare_features(df_sub, lags=[2, 4])
        
        if len(df_sub) < 10:
            continue
            
        target = 'average_salary'
        lag_features = ['lag_2', 'lag_4']
        
        base_features = ['gdp_mln_usd', 'exchange_rate_uah_usd', 'google_search_for_data_science', 
                         'final_consumption_expenditure', 'delta_gdp_mln_usd', 'percent_delta_gdp_mln_usd',
                         'consumer price index', 'average_salary_usd', 'delta_average_salary_usd',
                         'percent_delta_average_salary_usd', 'year', 'half', 'cleaned']
        
        # Перевірка наявності колонок
        available_features = [f for f in base_features if f in df_sub.columns]
        features = available_features + lag_features
        
        if not feature_columns:  # Зберігаємо список ознак
            feature_columns = features
        
        X = df_sub[features]
        y = df_sub[target]
        
        # Тренування моделі
        model = XGBRegressor(random_state=42, n_estimators=100)
        model.fit(X, y)
        
        models[seniority] = model
        print(f"✅ Модель для {seniority} натренована")
    
    return f"Натреновано {len(models)} моделей для рівнів: {', '.join(models.keys())}"

def predict_salary(seniority, gdp_mln_usd, exchange_rate, google_search, consumption_expenditure,
                  delta_gdp, percent_delta_gdp, cpi, avg_salary_usd, delta_salary_usd,
                  percent_delta_salary_usd, year, half, cleaned, lag_2, lag_4):
    """Передбачення зарплати на основі вхідних параметрів"""
    
    global models, feature_columns
    
    if not models:
        return "❌ Моделі не натреновані. Спочатку ініціалізуйте додаток."
    
    if seniority not in models:
        available = ', '.join(models.keys())
        return f"❌ Модель для '{seniority}' не знайдена. Доступні: {available}"
    
    try:
        # Створення DataFrame з вхідними даними
        input_data = {
            'gdp_mln_usd': [gdp_mln_usd],
            'exchange_rate_uah_usd': [exchange_rate],
            'google_search_for_data_science': [google_search],
            'final_consumption_expenditure': [consumption_expenditure],
            'delta_gdp_mln_usd': [delta_gdp],
            'percent_delta_gdp_mln_usd': [percent_delta_gdp],
            'consumer price index': [cpi],
            'average_salary_usd': [avg_salary_usd],
            'delta_average_salary_usd': [delta_salary_usd],
            'percent_delta_average_salary_usd': [percent_delta_salary_usd],
            'year': [year],
            'half': [half],
            'cleaned': [cleaned],
            'lag_2': [lag_2],
            'lag_4': [lag_4]
        }
        
        df_input = pd.DataFrame(input_data)
        
        # Перевірка наявності всіх потрібних ознак
        missing_features = [col for col in feature_columns if col not in df_input.columns]
        if missing_features:
            return f"❌ Відсутні ознаки: {', '.join(missing_features)}"
        
        # Передбачення
        model = models[seniority]
        X_input = df_input[feature_columns]
        prediction = model.predict(X_input)[0]
        
        # Додаткові розрахунки
        salary_usd = prediction / exchange_rate
        
        result = f"""
🎯 **Прогноз зарплати для {seniority.upper()}:**

💰 **Прогнозована зарплата:** {prediction:,.0f} UAH
💵 **В доларах США:** ${salary_usd:,.0f} USD
📊 **Курс долара:** {exchange_rate:.2f} UAH/USD

📈 **Вхідні параметри:**
• GDP: {gdp_mln_usd:,} млн USD
• Пошуки "data science": {google_search}
• Споживчі витрати: {consumption_expenditure:,} млн UAH
• Індекс споживчих цін: {cpi:.1f}
• Рік/півріччя: {year}/{half}
• Лаг 2 періоди: {lag_2:,.0f} UAH
• Лаг 4 періоди: {lag_4:,.0f} UAH
"""
        
        return result
        
    except Exception as e:
        return f"❌ Помилка при передбаченні: {str(e)}"

def load_custom_model(csv_file):
    """Завантаження власної моделі з CSV файлу"""
    if csv_file is None:
        return "❌ Файл не завантажено"
    
    try:
        result = train_initial_models(csv_file.name)
        return f"✅ {result}"
    except Exception as e:
        return f"❌ Помилка завантаження: {str(e)}"

# Створення Gradio інтерфейсу
def create_gradio_app():
    """Створення Gradio додатку для передбачення зарплат"""
    
    with gr.Blocks(title="💰 Salary Predictor", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 💰 Прогнозування зарплат в IT
        
        Введіть економічні параметри та отримайте прогноз зарплати для різних рівнів seniority
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔧 Ініціалізація моделі")
                init_btn = gr.Button("🚀 Ініціалізувати з демо-даними", variant="primary")
                
                gr.Markdown("### 📁 Або завантажте власні дані")
                csv_upload = gr.File(label="CSV файл", file_types=[".csv"])
                load_btn = gr.Button("📊 Тренувати на власних даних")
                
                init_status = gr.Textbox(label="Статус", interactive=False)
        
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
                
                gdp = gr.Number(label="GDP (млн USD)", value=180000)
                exchange_rate = gr.Number(label="Курс UAH/USD", value=27.5)
                google_search = gr.Number(label="Google пошуки 'data science'", value=75, precision=0)
                consumption = gr.Number(label="Споживчі витрати (млн UAH)", value=150000)
                
                with gr.Row():
                    delta_gdp = gr.Number(label="Δ GDP", value=0)
                    percent_delta_gdp = gr.Number(label="% Δ GDP", value=2.5)
                
                cpi = gr.Number(label="Індекс споживчих цін", value=110)
                
                with gr.Row():
                    avg_salary_usd = gr.Number(label="Сер. зарплата (USD)", value=1200)
                    delta_salary_usd = gr.Number(label="Δ Зарплата (USD)", value=50)
                
                percent_delta_salary = gr.Number(label="% Δ Зарплата (USD)", value=4)
                cleaned = gr.Dropdown(choices=[0, 1], label="Очищені дані", value=1)
                
                gr.Markdown("### 🔄 Лаги (попередні зарплати)")
                lag_2 = gr.Number(label="Зарплата 2 періоди назад (UAH)", value=35000)
                lag_4 = gr.Number(label="Зарплата 4 періоди назад (UAH)", value=33000)
                
                predict_btn = gr.Button("🎯 Передбачити зарплату", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📈 Результат прогнозу")
                prediction_output = gr.Markdown(value="Введіть параметри та натисніть 'Передбачити зарплату'")
        
        # Обробники подій
        init_btn.click(
            fn=lambda: train_initial_models(),
            outputs=init_status
        )
        
        load_btn.click(
            fn=load_custom_model,
            inputs=csv_upload,
            outputs=init_status
        )
        
        predict_btn.click(
            fn=predict_salary,
            inputs=[seniority, gdp, exchange_rate, google_search, consumption,
                   delta_gdp, percent_delta_gdp, cpi, avg_salary_usd, delta_salary_usd,
                   percent_delta_salary, year, half, cleaned, lag_2, lag_4],
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

# Запуск додатку
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        share=True,
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True
    )