

FEATURE_COLUMNS = [
    "Age",
    "BMI",
    "Blood_Pressure",
    "Cholesterol",
    "Glucose_Level",
    "Heart_Rate",
    "Sleep_Hours",
    "Exercise_Hours",
    "Water_Intake",
    "Stress_Level",
    "Smoking",
    "Alcohol",
    "Diet",
    "MentalHealth",
    "PhysicalActivity",
    "MedicalHistory",
    "Allergies",
    "Diet_Type__Vegan",
    "Diet_Type__Vegetarian",
    "Blood_Group_AB",
    "Blood_Group_B",
    "Blood_Group_O",
]

TARGET_COLUMN = "Target"


LEVEL_LABELS = {0: "Low / None", 1: "Moderate", 2: "High / Frequent"}


def risk_label(pred: int) -> str:
    return "High Risk" if pred == 1 else "Low Risk"
