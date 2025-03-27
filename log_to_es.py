from elasticsearch import Elasticsearch

# Connect to Elasticsearch (Default port is 9200)
es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "password"))

def log_alert(ip, score, predictions):
    doc = {"ip": ip, "anomaly_score": score, "model_predictions": predictions}
    es.index(index="alerts", document=doc)

# Example usage
log_alert("192.168.1.1", -0.85, [-1, -1, 1])

"""was unable to integrate this with the app.py due to some errors while trying to run elasticsearch
abd we were unable to connect to the port 9200"""