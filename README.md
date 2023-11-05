# Uczenie ciągłe z wykorzystaniem śnienia
Celem projektu będzie opracowanie metody generacji syntetycznych obrazów dla ustawienia uczenia ciągłego bez danych (data free incremental learning) przy pomocy mechanizmu śnienia. Praca będzie obejmować przegląd obecnie stosowanych metod oraz propozycje zabiegów architektonicznych mających na celu zwiększenie jakości generowanych obrazów.

[Project status](https://trello.com/b/tgBC6V52/praca-magisterska)  
[Experiment Runs](https://community.cloud.databricks.com/?o=5755659783198440#mlflow/experiments/55508159745560)

## Prepare environment
- Install Python version [3.10](https://www.python.org/downloads/release/python-31010/)
- Create a virtual environment `python -m venv .`
- Go to created environment `venv\Scripts\activate`
- Install packages `pip install -r requirements.txt`
- Create account on `https://community.cloud.databricks.com`
- Configure databricks access `databricks configure --host https://community.cloud.databricks.com/`
- Change parameters in config file if needed
- Run `python main.py`