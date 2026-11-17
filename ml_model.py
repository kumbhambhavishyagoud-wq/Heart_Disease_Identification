import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

# These seven fields match the reference predictor page.
INPUT_COLUMNS=["cp","trestbps","chol","fbs","restecg","thalach","exang"]
DISPLAY_FIELDS=[
    {"name":"cp","label":"Chest Pain Type","kind":"select","options":[("0","Typical Angina"),("1","Atypical Angina"),("2","Non-anginal Pain"),("3","Asymptomatic")]},
    {"name":"trestbps","label":"Resting BP (mm Hg)","kind":"number","placeholder":"e.g. 120","min":"70","max":"250"},
    {"name":"chol","label":"Serum Cholesterol (mg/dl)","kind":"number","placeholder":"e.g. 220","min":"80","max":"700"},
    {"name":"fbs","label":"Fasting Blood Sugar","kind":"select","options":[("0","Fasting Blood Sugar ≤ 120 mg/dl"),("1","Fasting Blood Sugar > 120 mg/dl")]},
    {"name":"restecg","label":"Resting Electro-cardiographic Result","kind":"select","options":[("0","Normal"),("1","ST-T wave abnormality"),("2","Left ventricular hypertrophy")]},
    {"name":"thalach","label":"Maximum Heart Rate Achieved","kind":"number","placeholder":"e.g. 150","min":"50","max":"250"},
    {"name":"exang","label":"Exercise Induced Angina","kind":"select","options":[("0","No"),("1","Yes")]}
]
TARGETS=["target","num","output","condition","heart_disease"]

def normalise_columns(df):
    df=df.copy()
    df.columns=[str(c).strip().lower().replace(" ","_").replace("-","_") for c in df.columns]
    # Common UCI Cleveland spelling aliases
    aliases={"resting_blood_pressure":"trestbps","serum_cholestoral":"chol",
             "serum_cholesterol":"chol","fasting_blood_sugar":"fbs",
             "resting_electrocardiographic_results":"restecg",
             "maximum_heart_rate_achieved":"thalach","exercise_induced_angina":"exang",
             "chest_pain_type":"cp","heart_disease":"target"}
    df=df.rename(columns=aliases)
    return df

def clean_dataset(source):
    df=normalise_columns(source)
    target=next((x for x in TARGETS if x in df.columns),None)
    if target is None: raise ValueError("Target column is missing. It must be named target, num, output, condition or heart_disease.")
    missing=[c for c in INPUT_COLUMNS if c not in df.columns]
    if missing: raise ValueError("Required columns missing: "+", ".join(missing))
    df=df[INPUT_COLUMNS+[target]].copy()
    for c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=[target])
    df[target]=(df[target]>0).astype(int)
    df=df.dropna(subset=INPUT_COLUMNS,how="all")
    if len(df)<30: raise ValueError("At least 30 usable patient records are required.")
    if df[target].nunique()!=2: raise ValueError("Target must contain both classes: 0 (no disease) and positive values (disease).")
    return df, target

def train_models(csv_path):
    source=pd.read_csv(csv_path)
    df,_=clean_dataset(source)
    X=df[INPUT_COLUMNS]
    y=df.iloc[:,-1]
    if y.value_counts().min()<2: raise ValueError("Need at least two examples of each target class.")
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    models={
      "Random Forest":RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced"),
      "Logistic Regression":LogisticRegression(max_iter=1500),
      "Decision Tree":DecisionTreeClassifier(max_depth=5,random_state=42),
      "Support Vector Machine":SVC(probability=True,random_state=42),
      "K-Nearest Neighbours":KNeighborsClassifier(n_neighbors=5)
    }
    results=[]; best=None; best_score=-1
    for name,clf in models.items():
        pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("classifier",clf)])
        start=time.perf_counter(); pipe.fit(Xtr,ytr); elapsed=time.perf_counter()-start
        yp=pipe.predict(Xte); score=accuracy_score(yte,yp)
        results.append({"algorithm":name,"accuracy":round(score*100,2),
          "precision":round(precision_score(yte,yp,zero_division=0)*100,2),
          "recall":round(recall_score(yte,yp,zero_division=0)*100,2),
          "f1":round(f1_score(yte,yp,zero_division=0)*100,2),
          "training_seconds":round(elapsed,4)})
        if name=="Random Forest": best=pipe; best_score=score
    report={"records":len(df),"input_features":INPUT_COLUMNS,"target":"binary disease outcome",
      "test_percent":20,"selected_model":"Random Forest","selected_model_accuracy":round(best_score*100,2),
      "results":sorted(results,key=lambda r:r["accuracy"],reverse=True)}
    return report,best
