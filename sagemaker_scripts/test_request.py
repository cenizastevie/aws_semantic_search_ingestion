import requests

with open("input/test_input.csv", "r", encoding="utf-8") as f:
    csv_data = f.read()

response = requests.post(
    "http://localhost:8080/invocations",
    data=csv_data,
    headers={"Content-Type": "text/csv"}
)

print("Prediction:", response.text)