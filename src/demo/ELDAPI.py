import json
from flask import Flask, request
import jpype

from src.eld import VectorBasedELD
jpype.startJVM(jpype.getDefaultJVMPath())
app = Flask(__name__)

module = VectorBasedELD(mode="demo")

@app.route("/eld/", methods=["POST"])
def eld():
	jpype.attachThreadToJVM()
	text = request.json
	return run(text)

def run(text):
	result = {}
	run_result = module(text)
	return json.dumps(run_result, ensure_ascii=False)
