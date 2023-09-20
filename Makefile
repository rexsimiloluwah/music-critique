.PHONY: 

run-app:
	poetry run streamlit run main.py 
run-tests:
	pytest