import requests
from pprint import PrettyPrinter
 
pp = PrettyPrinter()
pp.pprint(requests.get("http://localhost:8501/v1/models/email-classification-model").json())