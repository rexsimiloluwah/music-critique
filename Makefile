.PHONY: 

run-app:
	poetry run streamlit run app/1_Music_Critique.py
run-tests:
	pytest --browser=chrome --headless