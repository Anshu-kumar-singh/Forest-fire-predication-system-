
import requests
import json

try:
    print("Sending request to http://localhost:8000/api/predict...")
    response = requests.post(
        "http://localhost:8000/api/predict", 
        json={"region_id": "california"},
        timeout=10
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except:
        print("Response Text:")
        print(response.text)
        
except Exception as e:
    print(f"Request failed: {e}")
