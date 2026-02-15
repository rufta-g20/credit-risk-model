from src.api.pydantic_models import CustomerFeatures

data = {
    "R_Min_Days": 5.0,
    "F_Count": 10.0,
    "M_Debit_Total": 5500.0,
    "M_Debit_Mean": 550.0,
    "M_Debit_Std": 150.0,
    "Hour_Mode": 14,
    "Channel_Mode": "ChannelId_3",
    "Product_Mode": "airtime"
}

try:
    external_data = CustomerFeatures(**data)
    print("✅ Pydantic Schema is valid!")
except Exception as e:
    print(f"❌ Schema Error: {e}")