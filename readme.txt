Sistem informatic pentru detectarea fraudelor financiare utilizând tehnici de învățare automată

Descrierea proiectului
Acest proiect reprezintă o soluție software bazată pe inteligența artificială pentru detectarea fraudelor cu carduri bancare. Sistemul utilizează algoritmul XGBoost și tehnica SMOTE pentru a gestiona dezechilibrul datelor și include o interfață web interactivă (dezvoltată cu Streamlit) pentru analiza tranzacțiilor și explicabilitatea deciziilor prin SHAP.

Adresa repository: https://github.com/IrasogAlex/Licenta-Irasog-Alexandru

Structura proiectului
Codul sursă este organizat modular.
data/raw/ - locația unde se va plasa baza de date brută (creditcard.csv).
models/ - folderul unde vor fi salvate automat obiectele serializate și modelul antrenat.
src/ - conține scripturile pentru procesarea datelor, antrenare, evaluare și interfața web.

Baza de date
Modelul este conceput pentru a fi antrenat pe setul de date public „Credit Card Fraud Detection”.
Descărcați baza de date de pe platforma Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Extrageți arhiva și plasează fișierul obținut, denumit creditcard.csv, în folderul data/raw/.

Pași de instalare
Pentru a pregăti mediul de lucru, instalează dependențele și bibliotecile necesare rulând următoarea comandă în terminal:

pip install -r requirements.txt

Pași de compilare și lansare a aplicației

1. Antrenarea modelului (Training)
Pentru a procesa datele brute, a aplica tehnica SMOTE și a antrena modelul XGBoost, va fi rulată următoarea comandă:

python src/train.py --data-path data/raw/creditcard.csv --model-path models/xgb_fraud_detector.joblib

2. Evaluarea performanțelor
Pentru a genera metricile de performanță și graficele specifice pe setul de date izolat pentru testare, utilizeazți:

python src/evaluate.py --data-path data/raw/creditcard.csv --model-path models/xgb_fraud_detector.joblib

3. Lansarea aplicației web
Pentru a porni tabloul de control interactiv și a utiliza funcționalitățile de predicție (în lot sau pentru tranzacții individuale), porniți serverul Streamlit cu comanda:

streamlit run src/app.py
